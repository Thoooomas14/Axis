import torch
from torch.utils.data import IterableDataset
import tensorflow as tf

# Force TensorFlow to use CPU only in all workers
# This prevents VRAM OOM and CUDA initialization errors in forked processes.
tf.config.set_visible_devices([], 'GPU')

# Limit TensorFlow memory growth to prevent unbounded RAM usage
try:
    # Disable TF's aggressive memory allocation
    from tensorflow.python.framework import config as tf_config
    tf_config.set_soft_device_placement(True)
except Exception:
    pass

# Force TF to run in eager mode without caching
tf.config.run_functions_eagerly(True)

import tensorflow_datasets as tfds
import numpy as np
from scipy.spatial.transform import Rotation as R
from .goal_oracle import GoalOracle
import gc

# Import SE(3) utilities for Lie group operations
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.rotation_utils import so3_log, so3_exp, se3_log, quaternion_to_matrix

class RTXStreamLoader(IterableDataset):
    """
    Streams RT-X datasets from GCS and yields sliding windows with dynamic goals.
    
    Episodes are processed continuously (no subtask splitting).
    Goal vector updates dynamically when gripper state changes.
    
    Args:
        loss_horizon: Number of future steps to predict (for endpoint loss).
                     Ensures windows end early enough to have GT for horizon steps.
    
    Supports 'fractal20220817_data' and 'droid' datasets.
    """
    def __init__(self, dataset_name, split='train', batch_size=1, window_size=8, 
                 loss_horizon=1, image_key=None, image_size=(128, 128), shuffle_buffer_size=1000, 
                 data_dir=None, repeat=True):
        self.dataset_name = dataset_name
        self.split = split
        self.batch_size = batch_size
        self.window_size = window_size
        self.loss_horizon = loss_horizon  # How many future steps to output
        self.image_key = image_key  # Explicit image key (e.g., 'exterior_image_1_left')
        self.image_size = image_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.data_dir = data_dir
        self.repeat = repeat
        
        self.oracle = GoalOracle(output_dim=64)
        
        # Episode counter for periodic dataset reset
        self._episode_count = 0
        self._max_episodes_before_reset = 50  # Reset every 50 episodes to reclaim memory
        
        # Initialize dataset builder to get info
        self.builder = None
        self.ds = None
        
        if self.data_dir and self.data_dir.startswith('gs://'):
            try:
                import tensorflow_io as tfio
            except ImportError:
                print("Warning: tensorflow_io not found, GCS access might fail.")
            
        if self.data_dir and self.data_dir.startswith('gs://') and 'fractal' in self.dataset_name:
            full_path = f"{self.data_dir}/{self.dataset_name}/0.1.0"
            self.builder = tfds.builder_from_directory(builder_dir=full_path)
        else:
            # Only try GCS if explicitly requested or if data_dir is None (default TFDS behavior)
            use_gcs = (self.data_dir is None) 
            self.builder = tfds.builder(self.dataset_name, data_dir=self.data_dir, try_gcs=use_gcs)
            
    def __len__(self):
        """Returns the number of episodes in the dataset."""
        if self.builder:
            return self.builder.info.splits[self.split].num_examples
        return 0
    
    def estimate_total_windows(self, avg_episode_length=100):
        """
        Estimate total windows in the dataset for progress tracking.
        
        Args:
            avg_episode_length: Assumed average steps per episode (default 100 for RTX)
            
        Returns:
            Estimated number of windows (samples) in the dataset
        """
        num_episodes = len(self)
        if num_episodes == 0:
            return None
        
        # Windows per episode = episode_length - window_size - loss_horizon + 1
        # (or 0 if episode too short)
        usable_steps = max(0, avg_episode_length - self.window_size - self.loss_horizon + 1)
        return num_episodes * usable_steps

    def _quat_to_rotvec(self, quat: np.ndarray) -> np.ndarray:
        """
        Convert quaternion [qx, qy, qz, qw] to rotation vector (axis-angle).
        Uses scipy for numpy arrays (GPU-free data loading).
        """
        # Normalize quaternion
        quat = quat / (np.linalg.norm(quat) + 1e-8)
        # scipy expects [x, y, z, w] format - same as ours
        return R.from_quat(quat).as_rotvec().astype(np.float32)
    
    def _get_pose(self, obs):
        """
        Extracts 7D minimal SE(3) pose: [φ_x, φ_y, φ_z, p_x, p_y, p_z, gripper].
        
        φ is the rotation vector (log map of SO(3)) - axis-angle representation.
        p is the translation vector.
        """
        pos = np.zeros(3, dtype=np.float32)
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # Identity
        
        # 1. Try to get Pose
        if 'base_pose_tool_reached' in obs:
            # Fractal: 7D [x, y, z, qx, qy, qz, qw]
            p7 = obs['base_pose_tool_reached'].numpy()
            pos = p7[:3]
            quat = p7[3:7]
        elif 'cartesian_position' in obs:
            # Droid: 6D [x, y, z, rx, ry, rz] (Euler XYZ)
            p6 = obs['cartesian_position'].numpy()
            if p6.shape[0] == 6:
                pos = p6[:3]
                euler = p6[3:]
                quat = R.from_euler('xyz', euler).as_quat()  # [qx, qy, qz, qw]
        elif 'ee_pose' in obs:
            p7 = obs['ee_pose'].numpy()
            pos = p7[:3]
            if p7.shape[0] >= 7:
                quat = p7[3:7]
        elif 'pose' in obs:
            p7 = obs['pose'].numpy()
            pos = p7[:3]
            if p7.shape[0] >= 7:
                quat = p7[3:7]
        
        # Convert quaternion to rotation vector
        rotvec = self._quat_to_rotvec(quat)
        
        # 2. Try to get Gripper
        g = 0.0
        if 'gripper_closed' in obs: 
            g = obs['gripper_closed'].numpy()
        elif 'gripper_position' in obs:
            g = obs['gripper_position'].numpy()
        elif 'gripper_state' in obs: 
            g = obs['gripper_state'].numpy()
            
        if not np.isscalar(g): 
            g = g.item() if g.size == 1 else g[0]
        
        # 3. Construct 7D minimal representation: [φ, p, gripper]
        pose7 = np.zeros(7, dtype=np.float32)
        pose7[:3] = rotvec    # Rotation vector
        pose7[3:6] = pos      # Translation
        pose7[6] = g          # Gripper
        return pose7
    
    def _compute_twist(self, pose_curr: np.ndarray, pose_next: np.ndarray) -> np.ndarray:
        """
        Compute ground truth twist: ξ = log(T_curr⁻¹ · T_next).
        
        The twist is the relative motion from pose_curr to pose_next
        expressed in the Lie algebra se(3).
        
        Args:
            pose_curr: 7D [φ, p, gripper] current pose
            pose_next: 7D [φ, p, gripper] next pose
            
        Returns:
            7D twist [ω_x, ω_y, ω_z, v_x, v_y, v_z, gripper_delta]
        """
        # Build SE(3) matrices from poses
        R_curr = R.from_rotvec(pose_curr[:3]).as_matrix()
        p_curr = pose_curr[3:6]
        T_curr = np.eye(4, dtype=np.float32)
        T_curr[:3, :3] = R_curr
        T_curr[:3, 3] = p_curr
        
        R_next = R.from_rotvec(pose_next[:3]).as_matrix()
        p_next = pose_next[3:6]
        T_next = np.eye(4, dtype=np.float32)
        T_next[:3, :3] = R_next
        T_next[:3, 3] = p_next
        
        # Compute relative transform: T_rel = T_curr⁻¹ · T_next
        T_curr_inv = np.eye(4, dtype=np.float32)
        T_curr_inv[:3, :3] = R_curr.T
        T_curr_inv[:3, 3] = -R_curr.T @ p_curr
        
        T_rel = T_curr_inv @ T_next
        
        # Extract twist using logarithmic map
        # Use PyTorch utilities via numpy conversion
        T_rel_torch = torch.tensor(T_rel, dtype=torch.float32).unsqueeze(0)
        xi = se3_log(T_rel_torch).squeeze(0).numpy()  # 6D twist
        
        # Gripper delta (or just use next gripper state)
        gripper_delta = pose_next[6] - pose_curr[6]
        
        # Combine to 7D
        twist7 = np.zeros(7, dtype=np.float32)
        twist7[:6] = xi
        twist7[6] = gripper_delta
        return twist7

    def _process_image(self, img):
        if img is None: return np.zeros((3, 128, 128), dtype=np.float32)
        img = tf.image.resize(img, self.image_size)
        img = tf.cast(img, tf.float32) / 255.0
        return tf.transpose(img, [2, 0, 1]).numpy() # CHW

    def _process_episode(self, episode):
        """
        Generates windows from a single episode (continuous, no subtask splitting).
        
        Goal vector updates dynamically when gripper state changes (subtask transitions).
        Requery is NOT yielded since it's now trained as model confidence from action loss.
        
        Args:
            episode: dict containing 'steps' and optionally 'language_instruction'
        Yields:
            dict: Windowed sample with dynamically updated goal
        """
        # 1. Extract Full Episode Data
        imgs = []
        props = []
        
        # Fallback image keys if explicit key not specified
        fallback_keys = ['image', 'exterior_image_1_left', 'wrist_image_left', 'exterior_image_2_left']
        image_keys = [self.image_key] if self.image_key else fallback_keys
        
        for s in episode['steps']:
            obs = s['observation']
            
            img = None
            for key in image_keys:
                if key in obs:
                    img = obs[key]
                    break
            
            if img is not None:
                processed_img = self._process_image(img)
                imgs.append(processed_img)
            else:
                imgs.append(np.zeros((3, 128, 128), dtype=np.float32))
                
            props.append(self._get_pose(obs))
        
        if not imgs or len(imgs) < self.window_size + 1:
            return  # Episode too short
        
        imgs = np.array(imgs)
        props = np.array(props)
        total_steps = len(imgs)
        
        # 2. Identify Subtask Boundaries (gripper changes) for goal updates
        grippers = props[:, 6]
        is_closed = grippers > 0.5
        # Find indices where gripper state changes
        gripper_changes = np.where(is_closed[1:] != is_closed[:-1])[0] + 1
        # Add episode start and end
        subtask_boundaries = np.concatenate([[0], gripper_changes, [total_steps]])
        
        # 3. Precompute goals for each subtask segment
        # Each segment has a goal based on its start/end poses and task type
        segment_goals = []
        for i in range(len(subtask_boundaries) - 1):
            seg_start = subtask_boundaries[i]
            seg_end = subtask_boundaries[i + 1] - 1  # Last frame of segment
            if seg_end <= seg_start:
                seg_end = seg_start
            
            start_pose = props[seg_start]
            end_pose = props[seg_end]
            
            # Infer task type from gripper change
            start_grip = start_pose[6]
            end_grip = end_pose[6]
            
            task_type = 0  # Default: Move
            if start_grip < 0.5 and end_grip > 0.5:
                task_type = 1  # Pick
            elif start_grip > 0.5 and end_grip < 0.5:
                task_type = 2  # Place
            
            goal = self.oracle.encode_goal(task_type, start_pose, end_pose)
            segment_goals.append({
                'start': seg_start,
                'end': subtask_boundaries[i + 1],  # Exclusive end
                'goal': goal
            })
        
        # Helper function to get goal for a given frame index
        def get_goal_for_frame(frame_idx):
            for seg in segment_goals:
                if seg['start'] <= frame_idx < seg['end']:
                    return seg['goal']
            # Fallback to last segment's goal
            return segment_goals[-1]['goal'] if segment_goals else self.oracle.encode_goal(0, props[0], props[-1])
        
        # 4. Sliding Window over entire episode
        # Ensure we have enough future frames for loss_horizon steps after the window
        # Window ends at w + window_size - 1, we need data up to w + window_size - 1 + loss_horizon
        num_windows = total_steps - self.window_size - self.loss_horizon + 1
        
        if num_windows <= 0:
            return  # Episode too short for this window + horizon configuration
        
        for w in range(num_windows):
            # Window frames: [w, w+window_size)
            w_imgs = imgs[w : w + self.window_size]
            w_props = props[w : w + self.window_size]
            
            # Goal is based on the LAST frame of the window (current context)
            current_frame = w + self.window_size - 1
            goal_emb = get_goal_for_frame(current_frame)
            
            # Starting pose: last frame of observation window
            start_idx = w + self.window_size - 1
            
            # Compute twist targets for FUTURE steps (after the window)
            # These are the twists the model should predict to reach future poses
            target_twists = []
            target_poses = []  # Direct target poses for endpoint loss
            
            for h in range(self.loss_horizon):
                # Current and next indices for this future step
                curr_idx = start_idx + h
                next_idx = curr_idx + 1
                
                # Compute twist from current to next (guaranteed to exist due to windowing)
                twist = self._compute_twist(props[curr_idx], props[next_idx])
                target_twists.append(twist)
                
                # Target pose at this horizon step
                target_poses.append(props[next_idx])
            
            yield {
                'images': torch.tensor(w_imgs, dtype=torch.float32),
                'proprio': torch.tensor(w_props, dtype=torch.float32),
                'goal': goal_emb,
                'actions': torch.tensor(np.array(target_twists), dtype=torch.float32),  # (H, 7) twists
                'target_poses': torch.tensor(np.array(target_poses), dtype=torch.float32),  # (H, 7) future poses
            }
            
        # Explicit cleanup to break reference cycles immediately
        del imgs
        del props
        del episode
        gc.collect()

    def __iter__(self):
        # Load dataset in streaming mode
        # Disable autocache to prevent TFDS from loading typically "small" datasets into RAM
        # which defeats the purpose of streaming huge robotics datasets.
        read_config = tfds.ReadConfig(try_autocache=False, add_tfds_id=False)
        
        if self.ds is None:
            # We must use the same read_config for both cases
            if self.data_dir and self.data_dir.startswith('gs://') and 'fractal' in self.dataset_name:
                self.ds = self.builder.as_dataset(split=self.split, shuffle_files=True, read_config=read_config)
            else:
                self.ds = self.builder.as_dataset(split=self.split, shuffle_files=True, read_config=read_config)
            
        ds = self.ds
            
        # Sharding for PyTorch DataLoader Workers
        # This ensures each worker processes a unique slice of the dataset
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # tf.data.Dataset.shard(num_shards, index)
            ds = ds.shard(num_shards=worker_info.num_workers, index=worker_info.id)
            
        # Repeat indefinitely ONLY if requested
        # Note: Repeat should generally come after shard to ensure we repeat the shard
        if self.repeat:
            ds = ds.repeat()
            
        # Only shuffle if buffer is significant (>1) to save RAM
        # Droid images are huge; buffering 10+ episodes can OOM 64GB RAM.
        if self.shuffle_buffer_size > 1:
            ds = ds.shuffle(self.shuffle_buffer_size)
        
        for episode in ds:
            # Increment episode counter
            self._episode_count += 1
            
            # Periodic dataset reset to break TF's internal buffers
            if self._episode_count >= self._max_episodes_before_reset:
                self._episode_count = 0
                # Force clear TF session and dataset
                tf.keras.backend.clear_session()
                del episode
                # CRITICAL: Reset self.ds to force fresh dataset creation
                self.ds = None
                gc.collect()
                # Break out to restart iteration from __iter__
                # This forces a fresh dataset load
                return
            
            yield from self._process_episode(episode)
            
            # Active GC after each episode to prevent leaks
            del episode
            gc.collect()
            
            # Anti-Fragmentation for Persistent Workers
            # Force glibc to release free memory back to OS (Linux only)
            import platform
            if platform.system() == 'Linux':
                try:
                    import ctypes
                    libc = ctypes.CDLL("libc.so.6")
                    libc.malloc_trim(0)
                except Exception:
                    pass  # Silently ignore if malloc_trim fails
            # On Windows, gc.collect() above is the best we can do

