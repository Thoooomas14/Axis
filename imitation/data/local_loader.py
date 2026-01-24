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
    
    def _compute_episode_twists(self, props: np.ndarray) -> np.ndarray:
        """
        Vectorized twist computation for entire episode using PyPose.
        props: (T, 13)
        Returns: (T-1, 7) - [twist(6), gripper_delta(1)]
        """
        # Extract components
        R_flat = props[:, :9].reshape(-1, 3, 3) # (T, 3, 3)
        p = props[:, 9:12] # (T, 3)
        gripper = props[:, 12:13] # (T, 1)
        
        # SVD Orthonormalization (Robustness)
        # We must perform this here as well to match training logic
        U, S, V = torch.svd(torch.tensor(R_flat, dtype=torch.float32))
        with torch.no_grad():
             det = torch.det(U @ V.transpose(-2, -1))
             diag = torch.ones_like(S)
             diag[:, -1] = det
             R_clean = U @ torch.diag_embed(diag) @ V.transpose(-2, -1)
        
        # Construct SE(3) matrices
        T_mats = torch.eye(4).unsqueeze(0).repeat(len(props), 1, 1)
        T_mats[:, :3, :3] = R_clean
        T_mats[:, :3, 3] = torch.tensor(p, dtype=torch.float32)
        
        # Convert to PyPose LieTensor
        T_pp = pp.mat2SE3(T_mats, check=False)
        
        # Compute relative transforms: T_i^{-1} @ T_{i+1}
        # Slice: Current [0 : -1], Next [1 : ]
        T_curr = T_pp[:-1]
        T_next = T_pp[1:]
        
        T_rel = T_curr.Inv() @ T_next
        xi = pp.Log(T_rel).tensor().numpy() # (T-1, 6)
        
        # Gripper delta
        g_curr = gripper[:-1]
        g_next = gripper[1:]
        g_delta = g_next - g_curr
        
        # Combine
        twists = np.concatenate([xi, g_delta], axis=1) # (T-1, 7)
        return twists

    def _process_episode(self, imgs: np.ndarray, props: np.ndarray, object_props: dict = None):
        """Generate training windows from an episode."""
        if len(imgs) < self.buffer_size:
            return
        
        # Segment into subtasks
        segments = self._segment_subtasks(props)
        
        # Pre-compute ALL twists for the episode (Vectorized)
        # Shape: (T-1, 7)
        all_twists = self._compute_episode_twists(props)
        
        # Generate windows
        # Ensure we have enough future frames for the loss horizon
        # twist index i corresponds to transition from frame i to i+1
        # we need twists starting from (w + window_size - 1) up to horizon
        num_windows = len(imgs) - self.window_size - self.loss_horizon + 1
        
        for w in range(num_windows):
            w_imgs = imgs[w : w + self.window_size]
            w_props = props[w : w + self.window_size]
            
            # Find subtask for current window's last frame
            current_frame_idx = w + self.window_size - 1
            seg = self._get_segment_for_frame(segments, current_frame_idx)
            
            if seg is None:
                continue
            
            # Compute goal (with object properties)
            goal_emb = self.oracle.encode_goal(
                seg['task_type'],
                seg['start_pose'],
                seg['end_pose'],
                object_props
            )
            
            # Get Targets
            # Twist at index t is step t -> t+1
            # We want twists starting from current_frame_idx
            twist_start = current_frame_idx
            twist_end = twist_start + self.loss_horizon
            
            target_twists = all_twists[twist_start : twist_end] # (H, 7)
            
            # Target poses are the poses REACHED by the twists
            # If twist[t] goes t -> t+1, the target pose is t+1
            # So targets are props from current+1 to current+horizon+1
            pose_start = current_frame_idx + 1
            pose_end = pose_start + self.loss_horizon
            target_poses = props[pose_start : pose_end] # (H, 13)
            
            yield {
                'images': torch.tensor(w_imgs, dtype=torch.float32) / 255.0,
                'proprio': torch.tensor(w_props, dtype=torch.float32),
                'goal': goal_emb,
                'actions': torch.tensor(target_twists, dtype=torch.float32),
                'target_poses': torch.tensor(target_poses, dtype=torch.float32),
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
