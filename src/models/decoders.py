import torch
import torch.nn as nn


class ActionDecoder(nn.Module):
    """
    Decodes a single token into an action chunk.
    Input: (B, EmbedDim) - last token (current state).
    Output: (B, ChunkSize, 7) - predicted future trajectory.
    """

    def __init__(self, input_dim=256, output_dim=7, hidden_dim=128, chunk_size=10):
        super().__init__()
        self.chunk_size = chunk_size
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, chunk_size * output_dim),
        )

    def forward(self, x):
        """
        Args:
            x: Single token (B, EmbedDim)
        Returns:
            actions: (B, ChunkSize, OutputDim)
        """
        B = x.shape[0]
        out = self.net(x)  # (B, ChunkSize * output_dim)
        return out.view(B, self.chunk_size, self.output_dim)


class SafeActionDecoder(nn.Module):
    """
    Wraps any action decoder with safety bounds.
    Clamps position and rotation deltas to prevent unsafe motions.
    """

    def __init__(self, base_decoder, max_pos_delta=10.0, max_rot_delta=10.0):
        super().__init__()
        self.base_decoder = base_decoder
        self.max_pos_delta = max_pos_delta
        self.max_rot_delta = max_rot_delta

    def forward(self, x):
        """
        Args:
            x: Input to base decoder
        Returns:
            actions: Safety-clamped actions with same shape as base decoder output
        """
        actions = self.base_decoder(x)

        # 7D Output: [Angular(3), Linear(3), Gripper(1)]

        # Helper to clamp parts
        def clamp_twist(act_tensor):
            # act_tensor: (..., 7)
            # Angular Velocity (0:3) -> Clamped by max_rot_delta
            ang = torch.clamp(
                act_tensor[..., 0:3], -self.max_rot_delta, self.max_rot_delta
            )
            # Linear Velocity (3:6) -> Clamped by max_pos_delta
            lin = torch.clamp(
                act_tensor[..., 3:6], -self.max_pos_delta, self.max_pos_delta
            )
            # Gripper (6:7) -> Unchanged
            grip = act_tensor[..., 6:7]
            return torch.cat([ang, lin, grip], dim=-1)

        if actions.dim() == 2:
            # (B, action_dim)
            return clamp_twist(actions)
        if actions.dim() == 3:
            # (B, T, action_dim)
            return clamp_twist(actions)

        raise ValueError(f"Unexpected action tensor shape: {actions.shape}")


class RequeryDecoder(nn.Module):
    """
    Predicts confidence for the current prediction.
    Input: (B, EmbedDim) - last token (current state).
    Output: (B, 1) - confidence logit.
    """

    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: Single token (B, EmbedDim)
        Returns:
            logit: (B, 1)
        """
        return self.net(x)
