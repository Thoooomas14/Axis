import tensorflow_io as tfio # Register GCS filesystem
import tensorflow as tf
import torch
from torch.utils.data import IterableDataset
import tensorflow_datasets as tfds
import numpy as np
from training.utils.gemini_mock import MockGeminiPlanner

class RTXDataset(IterableDataset):
    """
    PyTorch IterableDataset for loading RT-X / RLDS datasets via TensorFlow Datasets.
    """
    def __init__(self, dataset_name, split='train', batch_size=1, image_size=(128, 128), shuffle_buffer_size=1000, data_dir=None, shuffle=True):
        self.dataset_name = dataset_name
        self.split = split
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.data_dir = data_dir
        self.shuffle = shuffle
        
        # Initialize Mocker
        self.mocker = MockGeminiPlanner()

        # Load the dataset builder
        try:
            # Force loading from local directory to avoid GCS checks
            if data_dir:
                self.builder = tfds.builder_from_directory(data_dir)
            else:
                self.builder = tfds.builder(dataset_name)
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            raise

    def _process_step(self, step):
        """
        Process a single step from the RLDS dataset.
        Extracts image, proprioception, and language instruction.
        """
        # ... (rest of _process_step is unused in __iter__ but kept for reference)
        pass

    def __iter__(self):
        """
        Yields batches of (images, proprio, action, goal_embs, mask)
        Shape: (B, T, ...)
        """
        # Create the dataset pipeline
        ds = self.builder.as_dataset(split=self.split, shuffle_files=self.shuffle)
        
        batch_images = []
        batch_proprio = []
        batch_action = []
        batch_goals = []
        batch_masks = []
        batch_requery = []
        
        max_len = 64 # Fixed sequence length for batching (truncate or pad)

        for episode in ds:
            # episode is a dict with 'steps' (Dataset) and 'language_instruction'
            
            # 1. Extract Instruction
            instr = ""
            if 'language_instruction' in episode:
                instr = episode['language_instruction'].decode('utf-8')
            
            # 2. Collect Steps
            steps = list(episode['steps']) # Convert to list of dicts
            episode_len = len(steps)
            
            if episode_len == 0:
                continue

            # 3. Extract Start/End Poses for Goal
            def get_pose(step_data):
                obs = step_data['observation']
                p = np.zeros(6, dtype=np.float32)
                if 'ee_pose' in obs: p = obs['ee_pose']
                elif 'pose' in obs: p = obs['pose']
                # Pad to 7D
                p7 = np.zeros(7, dtype=np.float32)
                p7[:6] = p[:6] if p.shape[0] >= 6 else np.pad(p, (0, 6-p.shape[0]))
                p7[6] = 1.0
                return p7

            # 4. Sub-Task Splitting Logic (Semantic)
            # We look for gripper state changes to identify Pick/Place events.
            # 'gripper_closed' is usually 0 (Open) or 1 (Closed).
            
            grippers = []
            for s in steps:
                obs = s['observation']
                if 'gripper_closed' in obs:
                    grippers.append(obs['gripper_closed'].numpy())
                else:
                    grippers.append(0.0) # Default open
            
            grippers = np.array(grippers).flatten()
            
            # Detect changes using threshold
            is_closed = grippers > 0.5
            changes = is_closed[1:] != is_closed[:-1]
            change_indices = np.where(changes)[0]
            
            split_idx = -1
            
            if len(change_indices) > 0:
                # Use the first significant change
                idx = change_indices[0]
                split_idx = idx + 1
                
                # Ensure split is within reasonable bounds
                if split_idx < 5 or split_idx > episode_len - 5:
                    split_idx = -1
            
            goal_embs_seq = [] 
            requery_labels_seq = [] 
            
            if split_idx != -1:
                # Valid Split Found
                
                # --- Phase 1 ---
                start_pose_1 = get_pose(steps[0])
                end_pose_1 = get_pose(steps[split_idx])
                goal_1 = self.mocker.encode_goal(start_pose=start_pose_1, end_pose=end_pose_1, instruction=instr)
                
                for t in range(split_idx + 1):
                    goal_embs_seq.append(goal_1)
                    requery_labels_seq.append(1.0 if t == split_idx else 0.0)
                    
                # --- Phase 2 ---
                start_pose_2 = get_pose(steps[split_idx + 1]) if split_idx + 1 < episode_len else end_pose_1
                end_pose_2 = get_pose(steps[-1])
                goal_2 = self.mocker.encode_goal(start_pose=start_pose_2, end_pose=end_pose_2, instruction=instr)
                
                for t in range(split_idx + 1, episode_len):
                    goal_embs_seq.append(goal_2)
                    requery_labels_seq.append(1.0 if t == episode_len - 1 else 0.0)
                    
            else:
                # Single Phase
                start_pose = get_pose(steps[0])
                end_pose = get_pose(steps[-1])
                goal = self.mocker.encode_goal(start_pose=start_pose, end_pose=end_pose, instruction=instr)
                
                for t in range(episode_len):
                    goal_embs_seq.append(goal)
                    requery_labels_seq.append(1.0 if t == episode_len - 1 else 0.0)

            # 5. Process Steps into Sequences
            imgs = []
            props = []
            acts = []
            
            for step_data in steps:
                # Image
                obs = step_data['observation']
                img = None
                for k in obs:
                    if 'image' in k: 
                        img = obs[k]
                        break
                if img is None: img = np.zeros(self.image_size + (3,), dtype=np.uint8)
                
                # Resize/Norm
                img = tf.image.resize(img, self.image_size)
                img = tf.cast(img, tf.float32) / 255.0
                img = tf.transpose(img, [2, 0, 1]).numpy() # CHW
                imgs.append(img)
                
                # Proprio
                p = get_pose(step_data)[:6]
                # Add gripper state
                g_state = 0.0
                if 'gripper_closed' in obs:
                    g_state = obs['gripper_closed'].numpy()
                elif 'gripper_state' in obs:
                    g_state = obs['gripper_state'].numpy()
                
                if not np.isscalar(g_state): g_state = g_state.item() if g_state.size == 1 else g_state[0]
                
                p7 = np.concatenate([p, [g_state]])
                props.append(p7)
                
                # Action
                a = np.zeros(7, dtype=np.float32)
                if 'action' in step_data:
                    raw_a = step_data['action']
                    if isinstance(raw_a, dict):
                        parts = []
                        if 'world_vector' in raw_a: parts.append(raw_a['world_vector'])
                        if 'rotation_delta' in raw_a: parts.append(raw_a['rotation_delta'])
                        if 'gripper_closedness_action' in raw_a:
                             g_act = raw_a['gripper_closedness_action']
                             # Ensure 1D
                             if len(g_act.shape) == 0: g_act = np.expand_dims(g_act, 0)
                             parts.append(g_act)
                        
                        if parts: a = np.concatenate(parts, axis=-1)
                    else:
                        a = raw_a
                if a.shape[0] < 7: a = np.pad(a, (0, 7-a.shape[0]))
                acts.append(a[:7])
                
            # 6. Pad/Truncate to max_len
            curr_len = len(imgs)
            
            # Convert lists to arrays first
            goal_embs_arr = np.array(goal_embs_seq) # (T, 77)
            requery_arr = np.array(requery_labels_seq).reshape(-1, 1) # (T, 1)
            
            if curr_len > max_len:
                # Truncate
                imgs = imgs[:max_len]
                props = props[:max_len]
                acts = acts[:max_len]
                goal_embs_arr = goal_embs_arr[:max_len]
                requery_arr = requery_arr[:max_len]
                mask = np.ones(max_len, dtype=np.float32)
            else:
                # Pad
                pad_len = max_len - curr_len
                imgs += [np.zeros_like(imgs[0])] * pad_len
                props += [np.zeros_like(props[0])] * pad_len
                acts += [np.zeros_like(acts[0])] * pad_len
                
                # Pad goals (repeat last goal or zero? Repeat last is safer for shape, but masked anyway)
                # Let's pad with zeros
                goal_pad = np.zeros((pad_len, 77), dtype=np.float32)
                goal_embs_arr = np.concatenate([goal_embs_arr, goal_pad], axis=0)
                
                # Pad requery (0)
                requery_pad = np.zeros((pad_len, 1), dtype=np.float32)
                requery_arr = np.concatenate([requery_arr, requery_pad], axis=0)
                
                mask = np.concatenate([np.ones(curr_len), np.zeros(pad_len)]).astype(np.float32)
                
            # Add to batch
            # So I should initialize `batch_requery` before the loop? No, the loop is inside `__iter__`.
            # I will assume `batch_requery` is NOT initialized and I need to handle it.
            # But `batch_images` etc are initialized BEFORE the loop.
            # I should probably replace the initialization block too or just add it.
            
            # Strategy: I will replace from `batch_images = []` down to the end of the method.
            
            pass # Placeholder for the logic above

