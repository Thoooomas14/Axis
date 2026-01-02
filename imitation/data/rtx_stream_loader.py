import torch
from torch.utils.data import IterableDataset
import tensorflow as tf
import tensorflow_datasets as tfds
# import tensorflow_io as tfio # Moved to __init__
import numpy as np
from scipy.spatial.transform import Rotation as R
from .goal_oracle import GoalOracle

class RTXStreamLoader(IterableDataset):
    """
    Streams RT-X datasets from GCS, splits them into subtasks, and yields sliding windows.
    Supports 'fractal20220817_data' and 'droid' datasets.
    """
    def __init__(self, dataset_name, split='train', batch_size=1, window_size=8, image_size=(128, 128), shuffle_buffer_size=1000, data_dir=None, repeat=True, max_ar_steps=1):
        self.dataset_name = dataset_name
        self.split = split
        self.batch_size = batch_size
        self.window_size = window_size
        self.image_size = image_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.data_dir = data_dir
        self.repeat = repeat
        self.max_ar_steps = max_ar_steps  # Number of sequential targets to yield
        
        self.oracle = GoalOracle(output_dim=64)
        
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

    # ... (methods _get_pose, _process_image, _process_episode remain unchanged) ...

    def _get_pose(self, obs):
        """Extracts 8D pose (3 Pos + 4 Quat + 1 Gripper) from observation."""
        p = np.zeros(7, dtype=np.float32) # Default 7D [x,y,z,qx,qy,qz,qw]
        
        # 1. Try to get Pose
        if 'base_pose_tool_reached' in obs:
            # Fractal: 7D [x, y, z, qx, qy, qz, qw]
            p = obs['base_pose_tool_reached'].numpy()
        elif 'cartesian_position' in obs:
            # Droid: 6D [x, y, z, rx, ry, rz] (Euler XYZ)
            p6 = obs['cartesian_position'].numpy()
            if p6.shape[0] == 6:
                pos = p6[:3]
                euler = p6[3:]
                # Convert Euler to Quat
                quat = R.from_euler('xyz', euler).as_quat() # [qx, qy, qz, qw]
                p = np.concatenate([pos, quat])
        elif 'ee_pose' in obs: 
            p = obs['ee_pose'].numpy()
        elif 'pose' in obs: 
            p = obs['pose'].numpy()
        
        # 2. Try to get Gripper
        g = 0.0
        if 'gripper_closed' in obs: 
            g = obs['gripper_closed'].numpy()
        elif 'gripper_position' in obs:
            # Droid: 1D float
            g = obs['gripper_position'].numpy()
        elif 'gripper_state' in obs: 
            g = obs['gripper_state'].numpy()
            
        if not np.isscalar(g): g = g.item() if g.size == 1 else g[0]
        
        # 3. Construct 8D
        p8 = np.zeros(8, dtype=np.float32)
        if p.shape[0] == 7:
            p8[:7] = p
        elif p.shape[0] == 6:
            # Fallback if conversion failed or wasn't done
            p8[:6] = p
        
        p8[7] = g
        return p8

    def _process_image(self, img):
        if img is None: return np.zeros((3, 128, 128), dtype=np.float32)
        img = tf.image.resize(img, self.image_size)
        img = tf.cast(img, tf.float32) / 255.0
        return tf.transpose(img, [2, 0, 1]).numpy() # CHW

    def _process_episode(self, episode):
        """
        Generates windows from a single episode.
        Args:
            episode: dict containing 'steps' and optionally 'language_instruction'
        Yields:
            dict: Windowed sample
        """
        # 1. Extract Full Episode Data without holding raw steps in memory
        imgs = []
        props = []
        
        # Image keys to check in order of preference
        image_keys = ['image', 'exterior_image_1_left', 'wrist_image_left', 'exterior_image_2_left']
        
        # Iterate directly over the TF dataset iterator to save memory
        for s in episode['steps']:
            obs = s['observation']
            
            # Handle image (search for first available key)
            img = None
            for key in image_keys:
                if key in obs:
                    img = obs[key]
                    break
            
            if img is not None:
                imgs.append(self._process_image(img))
            else:
                imgs.append(np.zeros((3, 128, 128), dtype=np.float32))
                
            props.append(self._get_pose(obs))
        
        if not imgs: return # Empty episode
        
        imgs = np.array(imgs)
        props = np.array(props)
        total_steps = len(imgs)
        
        # 2. Identify Subtasks (Gripper Changes)
        grippers = props[:, 7]
        is_closed = grippers > 0.5
        changes = np.where(is_closed[1:] != is_closed[:-1])[0] + 1
        
        split_indices = [0] + list(changes) + [total_steps]
        
        # 3. Process Subtasks
        for i in range(len(split_indices) - 1):
            start_idx = split_indices[i]
            end_idx = split_indices[i+1]
            
            # Minimum length check (for the core subtask, excluding lookback)
            if end_idx - start_idx < 2: continue
            
            # Include lookback from previous subtask for temporal context
            # This allows the model to see the approach motion before a pick/place
            lookback = min(start_idx, self.window_size - 1)
            lookback_start = start_idx - lookback
            
            subtask_imgs = imgs[lookback_start:end_idx]
            subtask_props = props[lookback_start:end_idx]
            
            # Track where the actual subtask starts within our extended array
            subtask_offset = lookback  # First `lookback` frames are context from previous subtask
            
            # Infer Task Type (from actual subtask, not lookback context)
            # 0: Move, 1: Pick, 2: Place
            actual_subtask_start = subtask_offset  # Index within subtask_props where actual subtask begins
            start_grip = subtask_props[actual_subtask_start][7]
            end_grip = subtask_props[-1][7]
            
            task_type = 0 # Default Move
            if start_grip < 0.5 and end_grip > 0.5:
                task_type = 1 # Pick
            elif start_grip > 0.5 and end_grip < 0.5:
                task_type = 2 # Place
            
            # Compute Goal for this subtask (using actual subtask bounds)
            start_pose = subtask_props[actual_subtask_start]
            end_pose = subtask_props[-1]
            goal_emb = self.oracle.encode_goal(task_type, start_pose, end_pose)
            
            # Windowing
            T = len(subtask_imgs)
            actual_subtask_len = T - subtask_offset  # Length of actual subtask (excluding lookback)
            
            # If subtask (including lookback) is shorter than window, pad it
            if T < self.window_size:
                pad_len = self.window_size - T
                # Pad with first frame at start (history)
                pad_imgs = np.tile(subtask_imgs[0:1], (pad_len, 1, 1, 1))
                pad_props = np.tile(subtask_props[0:1], (pad_len, 1))
                
                subtask_imgs = np.concatenate([pad_imgs, subtask_imgs], axis=0)
                subtask_props = np.concatenate([pad_props, subtask_props], axis=0)
                # Adjust offset to account for padding
                actual_subtask_start += pad_len
                T = len(subtask_imgs)
            
            # Sliding Window Logic
            num_windows = T - self.window_size + 1
            for w in range(num_windows):
                # Window indices: w to w+W
                w_imgs = subtask_imgs[w : w+self.window_size]
                w_props = subtask_props[w : w+self.window_size]
                
                # Target is the step AFTER the window.
                # For AR training, we need N sequential targets
                target_poses = []
                requery_flags = []
                
                for k in range(self.max_ar_steps):
                    target_idx = w + self.window_size + k
                    if target_idx < T:
                        target_poses.append(subtask_props[target_idx])
                        requery_flags.append(0.0)
                    else:
                        # End of subtask - use last pose and signal requery
                        target_poses.append(subtask_props[-1])
                        requery_flags.append(1.0)
                
                yield {
                    'images': torch.tensor(w_imgs, dtype=torch.float32),
                    'proprio': torch.tensor(w_props, dtype=torch.float32),
                    'goal': goal_emb,
                    'actions': torch.tensor(np.array(target_poses), dtype=torch.float32),  # (N, 8)
                    'requery': torch.tensor([requery_flags[0]], dtype=torch.float32)  # Use first step's requery for now
                }

    def __iter__(self):
        # Load dataset in streaming mode
        if self.ds is None:
            if self.data_dir and self.data_dir.startswith('gs://') and 'fractal' in self.dataset_name:
                self.ds = self.builder.as_dataset(split=self.split, shuffle_files=True)
            else:
                self.ds = self.builder.as_dataset(split=self.split, shuffle_files=True)
            
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
            yield from self._process_episode(episode)

