"""
Local Data Loader

Load preprocessed episodes from HDF5 and yield training windows.
Supports any RTX dataset (DROID, Fractal, etc.) that was preprocessed with preprocessor.py.
All derived computations (windowing, subtask segmentation, goals, twists) happen here.

Output format matches RTXStreamLoader exactly.
"""

import os
import sys
import random

import torch
from torch.utils.data import IterableDataset
import numpy as np
import h5py
from scipy.spatial.transform import Rotation as R

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pypose as pp
from .goal_oracle import GoalOracle


class LocalDataLoader(IterableDataset):
    """
    Load preprocessed episodes from HDF5.
    
    Supports any dataset preprocessed with preprocessor.py (DROID, Fractal, etc.).
    Performs windowing, subtask segmentation, goal computation, and twist
    computation at load time for maximum flexibility.
    
    Output format matches RTXStreamLoader exactly.
    """
    
    def __init__(
        self,
        data_path: str,
        window_size: int = 8,
        loss_horizon: int = 1,
        shuffle: bool = True,
        repeat: bool = True,
        split_start: float = 0.0,
        split_end: float = 1.0,
    ):
        """
        Args:
            data_path: Path to preprocessed HDF5 file
            window_size: Number of frames per training window
            loss_horizon: Number of future frames to predict
            shuffle: Whether to shuffle episodes
            repeat: Whether to repeat dataset infinitely
            split_start: Start of episode range (0.0 = first episode)
            split_end: End of episode range (1.0 = last episode)
            
        Example splits:
            Train: split_start=0.0, split_end=0.95 (first 95%)
            Val:   split_start=0.95, split_end=1.0 (last 5%)
        """
        self.data_path = data_path
        self.window_size = window_size
        self.loss_horizon = loss_horizon
        self.shuffle = shuffle
        self.repeat = repeat
        self.split_start = split_start
        self.split_end = split_end
        
        self.oracle = GoalOracle(output_dim=38)
        
        # Get episode keys, metadata, and apply split
        with h5py.File(data_path, 'r') as f:
            all_keys = sorted([k for k in f.keys() if k.startswith('episode_')])
            total = len(all_keys)
            
            start_idx = int(total * split_start)
            end_idx = int(total * split_end)
            
            self.episode_keys = all_keys[start_idx:end_idx]
            self.num_episodes = len(self.episode_keys)
            
            # Load metadata from preprocessor
            self.total_frames = f.attrs.get('total_frames', 0)
            self.avg_episode_length = f.attrs.get('avg_episode_length', 100.0)
            self.dataset_name = f.attrs.get('dataset', 'unknown')
        
        self.buffer_size = window_size + loss_horizon
    
    def estimate_total_windows(self) -> int:
        """Estimate total training windows for progress tracking."""
        windows_per_episode = max(1, self.avg_episode_length - self.buffer_size + 1)
        return int(self.num_episodes * windows_per_episode)
    
    def __len__(self):
        return self.num_episodes
    
    def _segment_subtasks(self, props: np.ndarray):
        """Segment episode into subtasks based on gripper changes (13D poses)."""
        grippers = props[:, 12]  # Gripper at index 12 in 13D pose
        is_closed = grippers > 0.5
        changes = np.where(is_closed[1:] != is_closed[:-1])[0] + 1
        boundaries = np.concatenate([[0], changes, [len(props)]])
        
        segments = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1] - 1
            if end < start:
                end = start
            
            start_pose = props[start]
            end_pose = props[end]
            
            # Determine task type (index 12 for gripper)
            if start_pose[12] < 0.5 and end_pose[12] > 0.5:
                task_type = 1  # Pick (gripper closing)
            elif start_pose[12] > 0.5 and end_pose[12] < 0.5:
                task_type = 2  # Place (gripper opening)
            else:
                task_type = 0  # Move (no change)
            
            segments.append({
                'start': start,
                'end': boundaries[i + 1],
                'start_pose': start_pose,
                'end_pose': end_pose,
                'task_type': task_type,
            })
        return segments
    
    def _get_segment_for_frame(self, segments, frame_idx):
        """Find which segment a frame belongs to."""
        for seg in segments:
            if seg['start'] <= frame_idx < seg['end']:
                return seg
        return segments[-1] if segments else None
    
    def _compute_twist(self, pose_curr: np.ndarray, pose_next: np.ndarray) -> np.ndarray:
        """Compute twist from 13D poses using PyPose: ξ = log(T_curr⁻¹ · T_next)."""
        R_curr = pose_curr[:9].reshape(3, 3)
        p_curr = pose_curr[9:12]
        T_curr = np.eye(4, dtype=np.float32)
        T_curr[:3, :3] = R_curr
        T_curr[:3, 3] = p_curr
        
        R_next = pose_next[:9].reshape(3, 3)
        p_next = pose_next[9:12]
        T_next = np.eye(4, dtype=np.float32)
        T_next[:3, :3] = R_next
        T_next[:3, 3] = p_next
        
        # Use PyPose for log map
        T_curr_pp = pp.mat2SE3(torch.tensor(T_curr, dtype=torch.float32).unsqueeze(0))
        T_next_pp = pp.mat2SE3(torch.tensor(T_next, dtype=torch.float32).unsqueeze(0))
        T_rel_pp = T_curr_pp.Inv() @ T_next_pp
        xi = pp.Log(T_rel_pp).tensor().squeeze(0).numpy()
        
        gripper_delta = pose_next[12] - pose_curr[12]
        
        twist7 = np.zeros(7, dtype=np.float32)
        twist7[:6] = xi
        twist7[6] = gripper_delta
        return twist7
    
    def _process_episode(self, imgs: np.ndarray, props: np.ndarray, object_props: dict = None):
        """Generate training windows from an episode."""
        if len(imgs) < self.buffer_size:
            return
        
        # Segment into subtasks
        segments = self._segment_subtasks(props)
        
        # Generate windows
        num_windows = len(imgs) - self.window_size - self.loss_horizon + 1
        
        for w in range(num_windows):
            w_imgs = imgs[w : w + self.window_size]
            w_props = props[w : w + self.window_size]
            
            # Find subtask for current window's last frame
            current_frame = w + self.window_size - 1
            seg = self._get_segment_for_frame(segments, current_frame)
            
            if seg is None:
                continue
            
            # Compute goal (with object properties)
            goal_emb = self.oracle.encode_goal(
                seg['task_type'],
                seg['start_pose'],
                seg['end_pose'],
                object_props
            )
            
            # Compute target twists and poses
            target_twists = []
            target_poses = []
            start_idx = w + self.window_size - 1
            
            for h in range(self.loss_horizon):
                curr_idx = start_idx + h
                next_idx = curr_idx + 1
                twist = self._compute_twist(props[curr_idx], props[next_idx])
                target_twists.append(twist)
                target_poses.append(props[next_idx])
            
            yield {
                'images': torch.tensor(w_imgs, dtype=torch.float32) / 255.0,
                'proprio': torch.tensor(w_props, dtype=torch.float32),
                'goal': goal_emb,
                'actions': torch.tensor(np.array(target_twists), dtype=torch.float32),
                'target_poses': torch.tensor(np.array(target_poses), dtype=torch.float32),
            }
    
    def __iter__(self):
        # Get worker info for sharding
        worker_info = torch.utils.data.get_worker_info()
        
        episode_keys = self.episode_keys.copy()
        
        # Shard across workers
        if worker_info is not None:
            per_worker = len(episode_keys) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else len(episode_keys)
            episode_keys = episode_keys[start:end]
        
        with h5py.File(self.data_path, 'r') as f:
            while True:
                if self.shuffle:
                    random.shuffle(episode_keys)
                
                for key in episode_keys:
                    ep = f[key]
                    imgs = ep['images'][:]  # (T, 3, H, W) uint8
                    props = ep['proprio'][:]  # (T, 13) float32
                    
                    # Load object properties if available (9D: size, color, shape)
                    object_props = None
                    if 'object_props' in ep:
                        obj_vec = ep['object_props'][:]  # (9,)
                        object_props = {
                            'size': obj_vec[:3],
                            'color': obj_vec[3:6],
                            'shape': obj_vec[6:9]
                        }
                    
                    yield from self._process_episode(imgs, props, object_props)
                
                if not self.repeat:
                    break
