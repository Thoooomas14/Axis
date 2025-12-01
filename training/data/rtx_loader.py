import tensorflow_io as tfio # Register GCS filesystem
import tensorflow as tf
import torch
from torch.utils.data import IterableDataset
import tensorflow_datasets as tfds
import numpy as np

class RTXDataset(IterableDataset):
    """
    PyTorch IterableDataset for loading RT-X / RLDS datasets via TensorFlow Datasets.
    """
    def __init__(self, dataset_name, split='train', batch_size=1, image_size=(128, 128), shuffle_buffer_size=1000, data_dir=None):
        self.dataset_name = dataset_name
        self.split = split
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.data_dir = data_dir

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
        # Extract Image (Camera 0 usually)
        # Adjust key based on specific dataset structure (e.g. 'image', 'observation/image')
        # For generic RT-X, we often look for 'observation' -> 'image'
        
        # Placeholder logic for standard RLDS structure
        # We assume 'observation' dict exists
        obs = step['observation']
        
        # Find the first image key
        image = None
        for key in obs.keys():
            if 'image' in key:
                image = obs[key]
                break
        
        if image is None:
            # Fallback or error
            image = tf.zeros(self.image_size + (3,), dtype=tf.uint8)
        else:
            image = tf.image.resize(image, self.image_size)
            image = tf.cast(image, tf.float32) / 255.0 # Normalize [0, 1]
            image = tf.transpose(image, [2, 0, 1]) # HWC -> CHW

        # Extract Proprioception (EE Pose)
        # We look for 'state' which might be joint pos, but we prefer EE pose if available.
        # In many RLDS datasets, 'observation/natural_language_instruction' is common.
        # For EE pose, we might look for 'observation/pose_6d' or similar.
        # Since we want to enforce 6D, we'll try to find it or placeholder it.
        # Note: Real implementation depends heavily on specific dataset keys.
        
        proprio = tf.zeros((6,), dtype=tf.float32) # Default dummy
        
        # Try to find EE pose in observation
        # Common keys: 'ee_pose', 'end_effector_pose', 'pose'
        # If not found, we might use joint pos if the user allows, but they requested EXCLUSIVELY EE pose.
        # So we'll stick to looking for it or zeroing.
        
        # Example logic for some datasets:
        if 'ee_pose' in obs:
            proprio = obs['ee_pose']
        elif 'pose' in obs:
            proprio = obs['pose']
            
        # Ensure 6D
        proprio = tf.reshape(proprio, [-1])
        # Use tf.pad to ensure at least 6 elements, then slice
        # This works in graph mode without explicit python conditionals on shape
        pad_amount = tf.maximum(0, 6 - tf.shape(proprio)[0])
        proprio = tf.pad(proprio, [[0, pad_amount]])
        proprio = proprio[:6]

        # Extract Action (EE Pose Target or Delta)
        # We want 6D output.
        action = tf.zeros((6,), dtype=tf.float32)
        
        if 'action' in step:
            raw_action = step['action']
            if isinstance(raw_action, dict):
                # Look for world_vector / rotation_delta
                # We need to handle this carefully in graph mode.
                # Assuming structure is consistent across dataset, keys should be present.
                # If keys are optional, we'd need more complex logic.
                # For fractal20220817, 'world_vector' and 'rotation_delta' are usually present.
                
                parts = []
                # Check keys existence is tricky in graph mode if structure varies.
                # But for a specific dataset, it's usually fixed.
                # We'll try to access them directly if we know the dataset.
                # Or use try/except block? No, not in graph.
                # We'll assume standard RT-X structure.
                
                # Note: In graph mode, dict keys are usually fixed.
                if 'world_vector' in raw_action: 
                    parts.append(raw_action['world_vector']) 
                if 'rotation_delta' in raw_action:
                    parts.append(raw_action['rotation_delta'])
                
                if len(parts) > 0:
                    action = tf.concat(parts, axis=-1)
            else:
                action = raw_action
                
        # Ensure 6D
        action = tf.reshape(action, [-1])
        pad_amount_action = tf.maximum(0, 6 - tf.shape(action)[0])
        action = tf.pad(action, [[0, pad_amount_action]])
        action = action[:6]


        # Extract Language Instruction
        # Usually in 'language_instruction' or 'task'
        if 'language_instruction' in step:
            lang = step['language_instruction']
        else:
            lang = tf.constant("", dtype=tf.string)

        return image, proprio, action, lang

    def __iter__(self):
        """
        Yields batches of (images, proprio, action, language_instructions)
        """
        # Create the dataset pipeline
        ds = self.builder.as_dataset(split=self.split, shuffle_files=True)
        
        # Flatten episodes to steps if needed, or iterate episodes
        # RLDS usually yields episodes. We want steps for training?
        # Or we want sequences.
        # For now, let's assume we flatten to steps for simple BC, 
        # or we keep episodes if we want to do sequence training.
        # Let's flatten for simplicity in this loader version.
        ds = ds.flat_map(lambda x: x['steps'])
        
        ds = ds.shuffle(self.shuffle_buffer_size)
        ds = ds.repeat() # Repeat indefinitely
        ds = ds.map(self._process_step, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        # Convert to numpy then torch
        iterator = ds.as_numpy_iterator()
        
        for batch in iterator:
            images, proprio, action, lang = batch
            
            # Convert to Torch
            images = torch.from_numpy(images)
            proprio = torch.from_numpy(proprio)
            action = torch.from_numpy(action)
            
            # Generate Mock Gemini Goal (77-dim)
            # Since we are iterating batches of steps, we don't have easy access to the full episode start/end here
            # without significant pipeline changes.
            # For now, we will generate the goal on-the-fly using the current step's info as a proxy,
            # or just random/default values to satisfy the interface.
            # In a real training setup, we'd likely pre-compute this or use a dataset that has episode boundaries.
            
            from training.utils.gemini_mock import MockGeminiPlanner
            mocker = MockGeminiPlanner()
            
            batch_goals = []
            for i in range(len(lang)):
                # Decode instruction bytes to string
                instr = lang[i].decode('utf-8') if isinstance(lang[i], bytes) else str(lang[i])
                
                # Use current proprio as start/target pose proxy
                current_pose = proprio[i].numpy()
                current_action = action[i].numpy()
                
                # Start Pose (7D)
                start_pose_7d = np.zeros(7)
                start_pose_7d[:6] = current_pose
                start_pose_7d[6] = 1.0 # qw

                # End Pose (7D) - Derive from Action
                # Action is usually a delta. We'll add it to the start pose position.
                # This ensures the "Goal" implies movement in the direction of the ground truth action.
                end_pose_7d = np.zeros(7)
                end_pose_7d[:3] = current_pose[:3] + current_action[:3] # Add position delta
                end_pose_7d[3:6] = current_pose[3:6] # Keep rotation same for simplicity (or add if simple)
                end_pose_7d[6] = 1.0
                
                # Mock goal
                goal_vec = mocker.encode_goal(start_pose=start_pose_7d, end_pose=end_pose_7d, instruction=instr)
                batch_goals.append(goal_vec)
            
            goal_embs = torch.tensor(np.array(batch_goals), dtype=torch.float32)
            
            yield images, proprio, action, goal_embs
