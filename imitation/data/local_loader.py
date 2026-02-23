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


def _local_episode_worker(data_path, buffer_size, mix_episodes, ram_usage_limit, result_queue, stop_event, episode_keys):
    """Background worker to read HDF5, process segments/twists, and return full mixed buffers."""
    import h5py
    import numpy as np
    import io
    import time
    import psutil
    from imitation.data.goal_oracle import extract_object_properties, GoalOracle
    from scipy.spatial.transform import Rotation as R
    import pypose as pp
    import os
    
    # Suppress TF logs in worker
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    oracle = GoalOracle(output_dim=38)
    
    # Copy helper methods for use in isolated worker
    def segment_subtasks(props: np.ndarray):
        grippers = props[:, 12]
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
            
            if start_pose[12] < 0.5 and end_pose[12] > 0.5: task_type = 1
            elif start_pose[12] > 0.5 and end_pose[12] < 0.5: task_type = 2
            else: task_type = 0
            
            segments.append({
                'start': start, 'end': boundaries[i + 1],
                'start_pose': start_pose, 'end_pose': end_pose, 'task_type': task_type
            })
        return segments

    def get_segment_for_frame(segments, frame_idx):
        for seg in segments:
            if seg['start'] <= frame_idx < seg['end']: return seg
        return segments[-1] if segments else None
        
    def compute_episode_twists(props: np.ndarray) -> np.ndarray:
        import torch
        R_flat = props[:, :9].reshape(-1, 3, 3)
        p = props[:, 9:12]
        gripper = props[:, 12:13]
        
        U, S, V = torch.svd(torch.tensor(R_flat, dtype=torch.float32))
        with torch.no_grad():
             det = torch.det(U @ V.transpose(-2, -1))
             diag = torch.ones_like(S)
             diag[:, -1] = det
             R_clean = U @ torch.diag_embed(diag) @ V.transpose(-2, -1)
        
        T_mats = torch.eye(4).unsqueeze(0).repeat(len(props), 1, 1)
        T_mats[:, :3, :3] = R_clean
        T_mats[:, :3, 3] = torch.tensor(p, dtype=torch.float32)
        
        T_pp = pp.mat2SE3(T_mats, check=False)
        T_curr = T_pp[:-1]
        T_next = T_pp[1:]
        
        T_rel = T_curr.Inv() @ T_next
        xi = pp.Log(T_rel).tensor().numpy()
        
        g_curr = gripper[:-1]
        g_next = gripper[1:]
        g_delta = g_next - g_curr
        
        return np.concatenate([xi, g_delta], axis=1)
        
    try:
        with h5py.File(data_path, 'r', rdcc_nbytes=4 * 1024 * 1024, libver='latest', swmr=True) as f:
            episode_buffer = []
            
            for key in episode_keys:
                if stop_event.is_set():
                    break
                    
                # RAM Check
                ram_used_gb = psutil.virtual_memory().used / (1024 ** 3)
                if ram_used_gb > ram_usage_limit:
                    if len(episode_buffer) == 0:
                        raise MemoryError(f"System RAM ({ram_used_gb:.1f}GB) exceeds limit ({ram_usage_limit}GB) before loading any episodes! Increase limit or free memory.")
                    while psutil.virtual_memory().used / (1024 ** 3) > ram_usage_limit:
                        if stop_event.is_set(): break
                        time.sleep(0.5)

                ep = f[key]
                imgs = ep['images'][:]
                props = ep['proprio'][:]
                
                props[:, 9:12] *= 1000.0 # Scale to mm
                
                if len(props) > 1:
                    pos_deltas = np.linalg.norm(np.diff(props[:, 9:12], axis=0), axis=1)
                    moving_mask = pos_deltas > 0.1
                    if moving_mask.any():
                        first_moving = np.argmax(moving_mask)
                        if first_moving > 0:
                            imgs = imgs[first_moving:]
                            props = props[first_moving:]

                if len(imgs) < buffer_size:
                    continue
                    
                object_props = None
                if 'language_instruction' in ep:
                    try:
                        instruction = str(ep['language_instruction'][()][0], 'utf-8')
                    except Exception:
                        instruction = ""
                    object_props = extract_object_properties(instruction)
                    
                object_props_vec = np.zeros(9, dtype=np.float32)
                if object_props:
                    object_props_vec = np.concatenate([
                        object_props['size'], object_props['color'], object_props['shape']
                    ]).astype(np.float32)

                twists = compute_episode_twists(props)
                segments = segment_subtasks(props)
                
                seq_len = len(imgs)
                goals = np.zeros((seq_len, 38), dtype=np.float32)
                subtask_ends = np.zeros((seq_len, 13), dtype=np.float32)
                
                for i in range(seq_len):
                    seg = get_segment_for_frame(segments, i)
                    if seg:
                        goals[i] = oracle.encode_goal(seg['task_type'], seg['start_pose'], seg['end_pose'], object_props)
                        subtask_ends[i] = seg['end_pose']
                    else:
                        goals[i] = oracle.encode_goal(0, props[i], props[i], object_props)
                        subtask_ends[i] = props[i]

                episode_buffer.append({
                    'images': imgs, 'proprio': props, 'twists': twists,
                    'goals': goals, 'subtask_ends': subtask_ends, 'object_props': object_props_vec
                })
                
                if len(episode_buffer) >= mix_episodes:
                    result_queue.put({'type': 'buffer', 'data': episode_buffer})
                    episode_buffer = []

            # Final flush
            if len(episode_buffer) > 0 and not stop_event.is_set():
                result_queue.put({'type': 'buffer', 'data': episode_buffer})
                
            result_queue.put({'type': 'done'})
            
    except Exception as e:
        import traceback
        result_queue.put({'type': 'error', 'data': f"{str(e)}\n{traceback.format_exc()}"})


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
        max_episodes: int = 0,
        mix_episodes: int = 8,
        ram_usage_limit: float = 12.0,
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
            max_episodes: Max episodes to use (0 = all). Useful for overfit testing.
            mix_episodes: Number of episodes to mix in RAM before yielding batches.
            ram_usage_limit: Maximum RAM usage (GB) before pausing background fetching.
            
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
        self.mix_episodes = mix_episodes
        self.ram_usage_limit = ram_usage_limit
        self.random_frames = False # Deprecated: unified index design handles this natively
        
        self.oracle = GoalOracle(output_dim=38)
        
        # Get episode keys, metadata, and apply split
        with h5py.File(data_path, 'r') as f:
            all_keys = sorted([k for k in f.keys() if k.startswith('episode_')])
            total = len(all_keys)
            
            start_idx = int(total * split_start)
            end_idx = int(total * split_end)
            
            self.episode_keys = all_keys[start_idx:end_idx]
            
            # Apply max_episodes limit (for overfit testing)
            if max_episodes > 0:
                self.episode_keys = self.episode_keys[:max_episodes]
            
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
    
    # Helper methods kept for backwards compatibility or other internal logic if ever called directly
    def _segment_subtasks(self, props: np.ndarray):
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
        for seg in segments:
            if seg['start'] <= frame_idx < seg['end']:
                return seg
        return segments[-1] if segments else None
    
    def _compute_episode_twists(self, props: np.ndarray) -> np.ndarray:
        R_flat = props[:, :9].reshape(-1, 3, 3) # (T, 3, 3)
        p = props[:, 9:12] # (T, 3)
        gripper = props[:, 12:13] # (T, 1)
        
        U, S, V = torch.svd(torch.tensor(R_flat, dtype=torch.float32))
        with torch.no_grad():
             det = torch.det(U @ V.transpose(-2, -1))
             diag = torch.ones_like(S)
             diag[:, -1] = det
             R_clean = U @ torch.diag_embed(diag) @ V.transpose(-2, -1)
        
        T_mats = torch.eye(4).unsqueeze(0).repeat(len(props), 1, 1)
        T_mats[:, :3, :3] = R_clean
        T_mats[:, :3, 3] = torch.tensor(p, dtype=torch.float32)
        
        T_pp = pp.mat2SE3(T_mats, check=False)
        T_curr = T_pp[:-1]
        T_next = T_pp[1:]
        
        T_rel = T_curr.Inv() @ T_next
        xi = pp.Log(T_rel).tensor().numpy() # (T-1, 6)
        
        g_curr = gripper[:-1]
        g_next = gripper[1:]
        g_delta = g_next - g_curr
        
        twists = np.concatenate([xi, g_delta], axis=1) # (T-1, 7)
        return twists

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        episode_keys = self.episode_keys.copy()
        
        if worker_info is not None:
            per_worker = int(np.ceil(len(episode_keys) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else len(episode_keys)
            episode_keys = episode_keys[start:end]
            
        ram_limit_gb = getattr(self, 'ram_usage_limit', 12.0)
        
        import multiprocessing as mp
        import queue
        
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue(maxsize=4) # Small queue, holds massive episode_buffers
        stop_event = ctx.Event()
        
        workers = []
        num_workers = worker_info.num_workers if worker_info else 1
        
        # To handle 'shuffle' and 'repeat', we build a master list of keys for the workers to iterate over
        # If repeat is True, the master loop resets. If False, workers exit when keys are exhausted.
        # But wait, IterableDataset iter restarts on epoch unless handled internally.
        
        try:
            while True:
                current_keys = episode_keys.copy()
                if self.shuffle:
                    random.shuffle(current_keys)
                    
                # Chunk keys identically for workers
                chunk_size = int(np.ceil(len(current_keys) / float(num_workers)))
                
                for i in range(num_workers):
                    keys_chunk = current_keys[i * chunk_size : (i + 1) * chunk_size]
                    if len(keys_chunk) == 0: continue
                    
                    p = ctx.Process(
                        target=_local_episode_worker,
                        args=(
                            self.data_path, self.buffer_size, self.mix_episodes,
                            ram_limit_gb, result_queue, stop_event, keys_chunk
                        )
                    )
                    p.daemon = True
                    p.start()
                    workers.append(p)
                
                active_workers = len(workers)
                
                while active_workers > 0:
                    try:
                        msg = result_queue.get(timeout=5.0)
                        
                        if msg['type'] == 'error':
                            print(f"Worker Error: {msg['data']}")
                            stop_event.set()
                            raise RuntimeError(msg['data'])
                            
                        elif msg['type'] == 'done':
                            active_workers -= 1
                            
                        elif msg['type'] == 'buffer':
                            episode_buffer = msg['data']
                            yield from self._yield_shuffled_buffer(episode_buffer)
                            
                    except queue.Empty:
                        # Check if all workers died silently
                        alive = sum([1 for p in workers if p.is_alive()])
                        if alive == 0 and active_workers > 0:
                            print("All workers died unexpectedly")
                            break
                            
                for p in workers:
                    if p.is_alive(): p.terminate()
                workers.clear()
                
                if not self.repeat:
                    break
                    
        except GeneratorExit:
            # Clean up processes on generator close
            stop_event.set()
            for p in workers:
                p.terminate()
        except Exception as e:
            print(f"Loader loop error: {e}")
            stop_event.set()
            raise
            raise e
            
    def _yield_shuffled_buffer(self, episode_buffer):
        """Build valid pointers directly to arrays and yield them uniformly shuffled."""
        valid_indices = []
        
        # 1. Map pointers
        for ep_id, ep_data in enumerate(episode_buffer):
            seq_len = len(ep_data['images'])
            max_start = seq_len - self.window_size - self.loss_horizon + 1
            
            if max_start <= 0: # Episode too short to form any window
                continue

            for start_idx in range(max_start):
                valid_indices.append((ep_id, start_idx))
                
        # 2. Global uniform shuffle
        if self.shuffle:
            random.shuffle(valid_indices)
            
        # 3. Dynamic Batch Slicing
        # The DataLoader expects to receive single items from __iter__ and then batches them.
        # However, RTXStreamLoader and LocalDataLoader historically yielded "batches" of size `window_size`
        # directly from __iter__. To maintain compatibility, we will yield a dictionary
        # where each tensor has a leading dimension of `window_size`.
        # This means we are effectively yielding one "window" at a time, but that window
        # itself contains `window_size` frames.

        # If random_frames was true, the old code yielded a batch of `window_size` random frames.
        # The new unified design means each yielded item is a "window" of `window_size` frames,
        # and if `self.shuffle` is true, these windows are drawn from random points across
        # mixed episodes.

        for ep_id, start_idx in valid_indices:
            ep = episode_buffer[ep_id]
            
            w_start = start_idx
            w_end = start_idx + self.window_size
            
            # Target twists and poses are for the future, starting from the last frame of the window
            # twist index i corresponds to transition from frame i to i+1
            # We need twists starting from (w_end - 1) up to loss_horizon
            twist_start = w_end - 1
            twist_end = twist_start + self.loss_horizon
            
            # Target poses are the poses REACHED by the twists
            # If twist[t] goes t -> t+1, the target pose is t+1
            # So targets are props from (w_end - 1) + 1 to (w_end - 1) + 1 + loss_horizon
            pose_start = w_end
            pose_end = pose_start + self.loss_horizon

            # Slicing creates Views, not full copies, massively reducing overhead
            yield {
                'images': torch.tensor(ep['images'][w_start : w_end], dtype=torch.float32) / 255.0, # (W, C, H, W)
                'proprio': torch.tensor(ep['proprio'][w_start : w_end], dtype=torch.float32),       # (W, 13)
                'goal': torch.tensor(ep['goals'][w_start : w_end], dtype=torch.float32),            # (W, 38)
                'actions': torch.tensor(ep['twists'][twist_start : twist_end], dtype=torch.float32), # (H, 7)
                'target_poses': torch.tensor(ep['proprio'][pose_start : pose_end], dtype=torch.float32), # (H, 13)
                'subtask_end_pose': torch.tensor(ep['subtask_ends'][w_start : w_end], dtype=torch.float32), # (W, 13)
                'object_props': torch.tensor(ep['object_props'], dtype=torch.float32),              # (9,)
            }
