import torch
from torch.utils.data import IterableDataset
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

class RTXDataset(IterableDataset):
    """
    PyTorch IterableDataset for loading RT-X / RLDS datasets via TensorFlow Datasets.
    """
    def __init__(self, dataset_name, split='train', batch_size=1, image_size=(128, 128), shuffle_buffer_size=1000):
        self.dataset_name = dataset_name
        self.split = split
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle_buffer_size = shuffle_buffer_size

        # Load the dataset builder
        try:
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
        if tf.shape(proprio)[0] != 6:
             # If we got something else (e.g. 7D with gripper, or 3D), handle it.
             # For now, just pad/slice to 6 for safety in this generic loader.
             proprio = tf.pad(proprio, [[0, max(0, 6 - tf.shape(proprio)[0])]])[:6]

        # Extract Action (EE Pose Target or Delta)
        # We want 6D output.
        action = tf.zeros((6,), dtype=tf.float32)
        
        if 'action' in step:
            raw_action = step['action']
            if isinstance(raw_action, dict):
                # Look for world vector / pose delta
                parts = []
                if 'world_vector' in raw_action: 
                    parts.append(raw_action['world_vector']) # Usually 3D
                if 'rotation_delta' in raw_action:
                    parts.append(raw_action['rotation_delta']) # Usually 3D
                
                if parts:
                    action = tf.concat(parts, axis=-1)
            else:
                action = raw_action
                
        # Ensure 6D
        action = tf.reshape(action, [-1])
        if tf.shape(action)[0] != 6:
             action = tf.pad(action, [[0, max(0, 6 - tf.shape(action)[0])]])[:6]


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
            # lang is numpy array of bytes/strings
            
            yield images, proprio, action, lang
