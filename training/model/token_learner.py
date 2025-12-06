import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenLearner(nn.Module):
    """
    Token Learner module (Ryoo et al., 2021).
    Learns to attend to specific spatial locations in the feature map to produce a fixed number of tokens.
    This acts as a bottleneck to compress high-dimensional visual features into a manageable sequence length.
    """
    def __init__(self, input_channels, num_tokens=8):
        super().__init__()
        self.num_tokens = num_tokens
        self.input_channels = input_channels

        # Learn 'num_tokens' spatial attention maps
        # We use a small Conv network to generate these maps from the input features
        self.attention_gen = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, groups=input_channels), # Depthwise
            nn.ReLU(),
            nn.Conv2d(input_channels, num_tokens, kernel_size=1) # Pointwise to num_tokens maps
        )

    def forward(self, x):
        """
        Args:
            x: Input feature map (B, C, H, W)
        Returns:
            tokens: (B, num_tokens, C)
        """
        B, C, H, W = x.shape
        
        # Generate attention maps: (B, num_tokens, H, W)
        attn_maps = self.attention_gen(x)
        
        # Softmax over spatial dimensions (H, W) to get valid probability distributions
        attn_maps = attn_maps.view(B, self.num_tokens, -1) # (B, K, H*W)
        attn_maps = F.softmax(attn_maps, dim=-1)
        attn_maps = attn_maps.view(B, self.num_tokens, H, W)

        # Apply attention to extract tokens
        # We want (B, K, C).
        # x is (B, C, H, W)
        # attn is (B, K, H, W)
        # We can use einsum or matrix multiplication.
        
        # Reshape x: (B, C, H*W)
        x_flat = x.view(B, C, -1)
        
        # Reshape attn: (B, K, H*W)
        attn_flat = attn_maps.view(B, self.num_tokens, -1)
        
        # Weighted sum: (B, K, H*W) @ (B, H*W, C) -> (B, K, C)
        # Transpose x_flat to (B, H*W, C)
        tokens = torch.bmm(attn_flat, x_flat.transpose(1, 2))
        
        return tokens
