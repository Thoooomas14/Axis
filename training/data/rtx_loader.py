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

        # Extract Proprioception
        # Usually 'state' or 'joint_positions'
        # We'll look for 'state'
        if 'state' in obs:
            proprio = obs['state']
        elif 'joint_positions' in obs:
            proprio = obs['joint_positions']
        else:
            proprio = tf.zeros((7,), dtype=tf.float32) # Dummy

        # Extract Language Instruction
        # Usually in 'language_instruction' or 'task'
        if 'language_instruction' in step:
            lang = step['language_instruction']
        else:
            lang = tf.constant("", dtype=tf.string)

        return image, proprio, lang

    def __iter__(self):
        """
        Yields batches of (images, proprio, language_instructions)
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
            images, proprio, lang = batch
            
            # Convert to Torch
            images = torch.from_numpy(images)
            proprio = torch.from_numpy(proprio)
            # lang is numpy array of bytes/strings
            
            yield images, proprio, lang
