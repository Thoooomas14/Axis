import torch
from torch.utils.data import IterableDataset
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_io as tfio # Required for GCS
import numpy as np
from scipy.spatial.transform import Rotation as R
from .goal_oracle import GoalOracle

class RTXStreamLoader(IterableDataset):
    """
    Streams RT-X datasets from GCS, splits them into subtasks, and yields sliding windows.
    Supports 'fractal20220817_data' and 'droid' datasets.
    """
    def __init__(self, dataset_name, split='train', batch_size=1, window_size=8, image_size=(128, 128), shuffle_buffer_size=1000, data_dir=None, repeat=True):
        self.dataset_name = dataset_name
        self.split = split
        self.batch_size = batch_size
        self.window_size = window_size
        self.image_size = image_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.data_dir = data_dir
        self.repeat = repeat
        
        self.oracle = GoalOracle(output_dim=64)

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
        steps = list(episode['steps'])
        if len(steps) == 0: return
        
        instr = ""
        if 'language_instruction' in episode:
            val = episode['language_instruction']
            if hasattr(val, 'numpy'): instr = val.numpy().decode('utf-8')
            else: instr = str(val)

        # 1. Extract Full Episode Data
        imgs = []
        props = []
        
        # Image keys to check in order of preference
        image_keys = ['image', 'exterior_image_1_left', 'wrist_image_left', 'exterior_image_2_left']
        
        for s in steps:
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
        
        imgs = np.array(imgs)
        props = np.array(props)
        
        # 2. Identify Subtasks (Gripper Changes)
        grippers = props[:, 7]
        is_closed = grippers > 0.5
        changes = np.where(is_closed[1:] != is_closed[:-1])[0] + 1
        
        split_indices = [0] + list(changes) + [len(steps)]
        
        # 3. Process Subtasks
        for i in range(len(split_indices) - 1):
            start_idx = split_indices[i]
            end_idx = split_indices[i+1]
            
            # Minimum length check
            if end_idx - start_idx < 2: continue
            
            subtask_imgs = imgs[start_idx:end_idx]
            subtask_props = props[start_idx:end_idx]
            
            # Infer Task Type
            # 0: Move, 1: Pick, 2: Place
            start_grip = subtask_props[0][7]
            end_grip = subtask_props[-1][7]
            
            task_type = 0 # Default Move
            if start_grip < 0.5 and end_grip > 0.5:
                task_type = 1 # Pick
            elif start_grip > 0.5 and end_grip < 0.5:
                task_type = 2 # Place
            
            # Compute Goal for this subtask
            start_pose = subtask_props[0]
            end_pose = subtask_props[-1]
            goal_emb = self.oracle.encode_goal(task_type, start_pose, end_pose)
            
            # Windowing
            T = len(subtask_imgs)
            
            # If subtask is shorter than window, pad it
            if T < self.window_size:
                pad_len = self.window_size - T
                # Pad with first frame at start (history)
                pad_imgs = np.tile(subtask_imgs[0:1], (pad_len, 1, 1, 1))
                pad_props = np.tile(subtask_props[0:1], (pad_len, 1))
                
                subtask_imgs = np.concatenate([pad_imgs, subtask_imgs], axis=0)
                subtask_props = np.concatenate([pad_props, subtask_props], axis=0)
                T = self.window_size # Now it's at least window size
            
            # Sliding Window Logic
            num_windows = T - self.window_size + 1
            for w in range(num_windows):
                # Window indices: w to w+W
                w_imgs = subtask_imgs[w : w+self.window_size]
                w_props = subtask_props[w : w+self.window_size]
                
                # Target is the step AFTER the window.
                if w + self.window_size < T:
                    target_pose = subtask_props[w + self.window_size]
                    requery = 0.0
                else:
                    # End of subtask
                    target_pose = subtask_props[-1]
                    requery = 1.0
                
                yield {
                    'images': torch.tensor(w_imgs, dtype=torch.float32),
                    'proprio': torch.tensor(w_props, dtype=torch.float32),
                    'goal': goal_emb,
                    'action': torch.tensor(target_pose, dtype=torch.float32),
                    'requery': torch.tensor([requery], dtype=torch.float32)
                }

    def __iter__(self):
        # Load dataset in streaming mode
        # Special handling for fractal on GCS to avoid recursion error
        if self.data_dir and self.data_dir.startswith('gs://') and 'fractal' in self.dataset_name:
            # Use builder_from_directory for GCS to avoid recursion error
            # Construct full path: gs://bucket/dataset_name/version
            # Note: We assume the data_dir points to the root containing the dataset folder
            # But builder_from_directory needs the specific dataset folder.
            # Let's try appending the dataset name.
            full_path = f"{self.data_dir}/{self.dataset_name}/0.1.0"
            builder = tfds.builder_from_directory(builder_dir=full_path)
            ds = builder.as_dataset(split=self.split, shuffle_files=True)
        else:
            # Fallback to standard load (works for droid and local)
            ds = tfds.load(
                self.dataset_name, 
                split=self.split, 
                shuffle_files=True, 
                try_gcs=True,
                data_dir=self.data_dir
            )
            
        # Repeat indefinitely ONLY if requested
        if self.repeat:
            ds = ds.repeat()
            
        ds = ds.shuffle(self.shuffle_buffer_size)
        
        for episode in ds:
            yield from self._process_episode(episode)

