import torch
import torch.nn as nn

class LatentQueue(nn.Module):
    """
    Manages a sliding window of latent states.
    """
    def __init__(self, queue_size, latent_dim, device='cpu'):
        super().__init__()
        self.queue_size = queue_size
        self.latent_dim = latent_dim
        self.device = device
        
        # Initialize queue with zeros or learnable parameters
        # Shape: (1, queue_size, latent_dim) - Batch size 1 for now, handled dynamically in forward
        self.register_buffer('queue', torch.zeros(1, queue_size, latent_dim))
        self.ptr = 0

    def reset(self, batch_size):
        """Resets the queue for a new episode."""
        self.queue = torch.zeros(batch_size, self.queue_size, self.latent_dim).to(self.queue.device)

    def enqueue(self, new_latent):
        """
        Adds a new latent state to the queue, removing the oldest.
        Args:
            new_latent: (B, latent_dim)
        """
        # Check for batch size mismatch
        B = new_latent.shape[0]
        if self.queue.shape[0] != B:
            if self.queue.shape[0] == 1:
                self.queue = self.queue.expand(B, -1, -1).clone()
            else:
                raise ValueError(f"Batch size mismatch: Queue {self.queue.shape[0]} vs Input {B}")

        # Shift everything to the right (discard oldest at end)
        # queue: (B, T, D)
        # We want to discard index T-1, and prepend new_latent at index 0
        # Order becomes: [Newest (t), t-1, t-2, ..., Oldest]
        self.queue = torch.cat([new_latent.unsqueeze(1), self.queue[:, :-1, :]], dim=1)

    def forward(self):
        """
        Returns the current queue state.
        Returns:
            (B, queue_size, latent_dim)
        """
        return self.queue

class LatentProjection(nn.Module):
    """
    Projects the raw latent state into the transformer's embedding dimension.
    """
    def __init__(self, latent_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)
