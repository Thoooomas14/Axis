import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights

class VisionEncoder(nn.Module):
    """
    Encodes RGB images using a pretrained ResNet-18 backbone.
    Outputs feature map (B, C, H, W).
    """
    def __init__(self, input_channels=3, feature_dim=256, pretrained=True):
        super().__init__()
        
        # Load ResNet-18
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        
        # If input channels != 3, adjust the first conv layer
        if input_channels != 3:
            backbone.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # We take layers up to layer4 (before avgpool/fc) to keep spatial features
        # ResNet layer channels: layer1=64, layer2=128, layer3=256, layer4=512
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        
        # Project 512 channels from layer4 to feature_dim
        self.proj = nn.Conv2d(512, feature_dim, kernel_size=1)
        self.bn_proj = nn.BatchNorm2d(feature_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: Images (B, C, H, W)
        Returns:
            Feature map (B, feature_dim, H_out, W_out)
        """
        x = self.backbone(x)
        x = self.act(self.bn_proj(self.proj(x)))
        return x

class ProprioEncoder(nn.Module):
    """
    Encodes robot proprioceptive state (End-Effector Pose) into a latent vector.
    Encodes robot proprioceptive state (End-Effector Pose) into a latent vector.
    Default input_dim=7 (3 Rotation Vector + 3 Translation + 1 Gripper).
    
    Args:
        input_dim: Dimension of proprioceptive input (default 7)
        output_dim: Dimension of output embedding
        hidden_dim: Dimension of hidden layers
        normalize: If True, apply running normalization using EMA statistics
    """
    def __init__(self, input_dim=7, output_dim=256, hidden_dim=128, normalize=True):
        super().__init__()
        self.input_dim = input_dim
        self.normalize = normalize
        
        # Running statistics buffers for normalization
        if normalize:
            self.register_buffer('running_mean', torch.zeros(input_dim))
            self.register_buffer('running_std', torch.ones(input_dim))
            self.register_buffer('count', torch.tensor(0, dtype=torch.long))
            self.ema_alpha = 0.1
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def update_stats(self, x):
        """
        Update running mean and std with EMA (alpha=0.1).
        Only call during training.
        
        Args:
            x: Input tensor of shape (B, input_dim) or (B, T, input_dim)
        """
        if not self.normalize:
            return
        
        # Flatten to (N, input_dim) if needed
        if x.dim() == 3:
            x = x.reshape(-1, x.size(-1))
        
        # Compute batch statistics
        batch_mean = x.mean(dim=0)
        batch_std = x.std(dim=0)
        
        # EMA update
        if self.count == 0:
            # First update: initialize with batch stats
            self.running_mean.copy_(batch_mean)
            self.running_std.copy_(batch_std.clamp(min=1e-6))
        else:
            # EMA: new = (1 - alpha) * old + alpha * batch
            self.running_mean.copy_(
                (1 - self.ema_alpha) * self.running_mean + self.ema_alpha * batch_mean
            )
            self.running_std.copy_(
                (1 - self.ema_alpha) * self.running_std + self.ema_alpha * batch_std.clamp(min=1e-6)
            )
        
        self.count.add_(1)

    def forward(self, x):
        """
        Args:
            x: Proprioceptive state (B, input_dim) or (B, T, input_dim)
        Returns:
            Encoded embedding of same batch shape with output_dim
        """
        # Apply normalization if enabled
        if self.normalize:
            # Update stats during training
            if self.training:
                self.update_stats(x)
            
            # Normalize: (x - mean) / std
            # Clamp std to avoid division by zero
            std_clamped = self.running_std.clamp(min=1e-6)
            x = (x - self.running_mean) / std_clamped
        
        return self.net(x)




class GoalEncoder(nn.Module):
    """
    Project semantic goal embeddings (from LLM/Gemini) into the model's dimension.
    """
    def __init__(self, input_dim=64, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)



