import torch
import torch.nn as nn

class ActionDecoder(nn.Module):
    """
    Decodes the transformer output into robot actions (Typcally 6D Twist + Gripper).
    Default output_dim=7: 3 angular vel + 3 linear vel + 1 gripper.
    """
    def __init__(self, input_dim=256, output_dim=7, hidden_dim=128):
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



class SafeActionDecoder(nn.Module):
    """
    Wraps any action decoder with safety bounds.
    Clamps position and rotation deltas to prevent unsafe motions.
    """
    def __init__(self, base_decoder, max_pos_delta=0.05, max_rot_delta=0.1):
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
            ang = torch.clamp(act_tensor[..., 0:3], -self.max_rot_delta, self.max_rot_delta)
            # Linear Velocity (3:6) -> Clamped by max_pos_delta
            lin = torch.clamp(act_tensor[..., 3:6], -self.max_pos_delta, self.max_pos_delta)
            # Gripper (6:7) -> Unchanged
            grip = act_tensor[..., 6:7]
            return torch.cat([ang, lin, grip], dim=-1)

        if actions.dim() == 2:
             # (B, action_dim)
             return clamp_twist(actions)
        elif actions.dim() == 3:
             # (B, T, action_dim)
             return clamp_twist(actions)
        else:
             raise ValueError(f"Unexpected action tensor shape: {actions.shape}")


class RequeryDecoder(nn.Module):
    """
    Predicts if the current sub-task is complete and a new goal is needed.
    Outputs a logit (use sigmoid for probability).
    Supports 2D (B, D) or 3D (B, T, D) inputs.
    """
    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        # Content-based attention pooling to aggregate chunk
        self.attn_proj = nn.Linear(embed_dim, 1)
        
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Input tokens (B, T, D)
        Returns:
            logit: (B, 1)
        """
        if x.dim() != 3:
            raise ValueError(f"RequeryDecoder expects 3D input (B, T, D), got {x.shape}")
             
        # (B, T, D) -> Attention Pooling -> (B, D)
        # 1. Compute attention scores for each step
        attn_logits = self.attn_proj(x) # (B, T, 1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        
        # 2. Weighted sum
        x_pooled = (x * attn_weights).sum(dim=1) # (B, D)
        return self.net(x_pooled)
