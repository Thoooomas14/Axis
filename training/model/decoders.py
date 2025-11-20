import torch
import torch.nn as nn

class ActionDecoder(nn.Module):
    """
    Decodes the transformer output into robot actions (e.g., joint positions).
    Usually attends to the last token or a specific 'action' token, 
    or we can just pool the output. 
    For simplicity, we'll assume we take the output corresponding to the *current* state tokens.
    """
    def __init__(self, embed_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, action_dim)
        )

    def forward(self, x):
        """
        Args:
            x: Transformer output token(s) (B, EmbedDim) - usually pooled or specific token
        """
        return self.net(x)

class LatentDecoder(nn.Module):
    """
    Predicts the next latent state from the transformer output.
    """
    def __init__(self, embed_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, latent_dim)
        )

    def forward(self, x):
        return self.net(x)
