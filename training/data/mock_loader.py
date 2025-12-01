import torch
from torch.utils.data import IterableDataset
import numpy as np

class MockDataset(IterableDataset):
    """
    Mock Dataset that yields random data matching the shapes of RTXDataset.
    Used for testing the training loop and performance benchmarking without real data.
    """
    def __init__(self, dataset_name='mock', split='train', batch_size=1, image_size=(128, 128), 
                 proprio_dim=6, action_dim=6, length=1000):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.image_size = image_size
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.length = length # Virtual length for the iterator

    def __iter__(self):
        """
        Yields batches of (images, proprio, action, language_instructions)
        """
        count = 0
        while count < self.length:
            # Generate random batch
            # Images: [B, C, H, W]
            images = torch.randn(self.batch_size, 3, *self.image_size)
            
            # Proprio: [B, proprio_dim]
            proprio = torch.randn(self.batch_size, self.proprio_dim)
            
            # Action: [B, action_dim]
            action = torch.randn(self.batch_size, self.action_dim)
            
            # Goal Embeddings: [B, 77]
            # Structured Gemini Goal
            goal_embs = torch.randn(self.batch_size, 77)
            
            yield images, proprio, action, goal_embs
            
            count += 1
