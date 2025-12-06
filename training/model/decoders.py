import torch
import torch.nn as nn

class ActionDecoder(nn.Module):
    """
    Decodes the transformer output into robot actions (End-Effector Pose/Delta).
    Default output_dim=6 (3 Pos + 3 Rot).
    """
    def __init__(self, input_dim=256, output_dim=6, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Args:
            x: Transformer output token(s) (B, EmbedDim)
        """
        return self.net(x)

class RequeryDecoder(nn.Module):
    """
    Predicts if the current sub-task is complete and a new goal is needed.
    Outputs a logit (use sigmoid for probability).
    """
    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)
