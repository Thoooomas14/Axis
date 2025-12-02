import torch
from torch.utils.data import IterableDataset
import numpy as np

class MockDataset(IterableDataset):
    """
    Mock Dataset that yields random data matching the shapes of RTXDataset.
    Used for testing the training loop and performance benchmarking without real data.
    """
    def __init__(self, dataset_name='mock', split='train', batch_size=1, image_size=(128, 128), 
                 proprio_dim=7, action_dim=7, length=1000):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.image_size = image_size
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.length = length # Virtual length for the iterator

    def __iter__(self):
        """
        Yields batches of (images, proprio, action, goal_embs, mask)
        Shape: (B, T, ...)
        """
        count = 0
        T = 10 # Sequence length
        
        while count < self.length:
            # Generate random batch
            # Images: [B, T, C, H, W]
            images = torch.randn(self.batch_size, T, 3, *self.image_size)
            
            # Proprio: [B, T, proprio_dim]
            proprio = torch.randn(self.batch_size, T, self.proprio_dim)
            
            # Action: [B, T, action_dim]
            action = torch.randn(self.batch_size, T, self.action_dim)
            
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
        Yields batches of (images, proprio, action, goal_embs, mask)
        Shape: (B, T, ...)
        """
        count = 0
        T = 10 # Sequence length
        
        while count < self.length:
            # Generate random batch
            # Images: [B, T, C, H, W]
            images = torch.randn(self.batch_size, T, 3, *self.image_size)
            
            # Proprio: [B, T, proprio_dim]
            proprio = torch.randn(self.batch_size, T, self.proprio_dim)
            
            # Action: [B, T, action_dim]
            action = torch.randn(self.batch_size, T, self.action_dim)
            
            # Goal Embeddings: [B, 77] (Shared for episode, but we might want to expand or keep as is. train.py expects [B, 77])
            # Wait, train.py expects goal_embs to be (B, 77) and expands it.
            goal_embs = torch.randn(self.batch_size, 77)
            
            # Mask: [B, T]
            mask = torch.ones(self.batch_size, T)
        
            # Requery Labels (B, T, 1)
            # Randomly set last step to 1
            requery_labels = torch.zeros(self.batch_size, T, 1)
            requery_labels[:, -1, :] = 1.0
            
            yield images, proprio, action, goal_embs, mask, requery_labels
            
            count += 1
