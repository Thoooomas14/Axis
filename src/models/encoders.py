import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights

class VisionEncoder(nn.Module):
    """
    Encodes RGB images using a pretrained ResNet-18 backbone.
    Outputs a 256D latent vector that explains the image content.
    Includes probe heads for pre-training (Reconstruction, Proprio, Object Props).
    """
    def __init__(self, input_channels=3, feature_dim=256, goal_dim=38, pretrained=True):
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
            backbone.maxpool, # -> /4
            backbone.layer1,  # -> /4
            backbone.layer2,  # -> /8
            backbone.layer3,  # -> /16
            backbone.layer4   # -> /32 (B, 512, H/32, W/32)
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- Cross Attention for Goal ---
        self.goal_proj = nn.Linear(goal_dim, 512)
        self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)
        
        # Project 512 channels from layer4 to feature_dim
        self.proj = nn.Linear(512, feature_dim)
        # We purposely do not use LayerNorm or ReLU here. 
        # LayerNorm forces the mean to be 0 across the latent vector, which
        # disables its ability to store global intensity (like overall brightness/color).

        # --- Pre-training Heads ---
        # 1. Image Reconstruction Decoder (Latent -> Image)
        # Reconstructs from 1D latent to force compression of "Environment/Obstacles"
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 256, kernel_size=4, stride=1, padding=0), # 1x1 -> 4x4
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(16, input_channels, kernel_size=4, stride=2, padding=1), # 64x64 -> 128x128
            nn.Sigmoid() # Pixel values [0, 1]
        )
        
        # 2. Proprioception Probe (Latent -> 13D)
        # "Where am I?"
        self.proprio_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 13)
        )
        
        # 3. Subtask End Pose Probe (Latent -> 13D)
        # "Where is the object/target?" (Proxy)
        self.end_pose_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 13)
        )
        
        # 4. Object Properties Probe (Latent -> 9D)
        # "What is it?" (Size, Color, Shape)
        self.obj_props_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )

    def forward(self, x, raw_goal=None, return_preds=False):
        """
        Args:
            x: Images (B, C, H, W)
            raw_goal: Optional (B, goal_dim) for spatial cross-attention (raw 38D vector)
            return_preds: If True, return reconstruction and probe predictions.
        Returns:
            latent: (B, feature_dim)
            preds: Dict of predictions (if return_preds=True)
        """
        # ImageNet normalization for the pretrained ResNet backbone.
        # Without this, the backbone sees a completely wrong input distribution.
        if x.size(1) == 3:
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            x_norm = (x - mean) / std
        else:
            x_norm = x

        features = self.backbone(x_norm) # (B, 512, H/32, W/32)
        
        if raw_goal is not None:
            B, C_f, H_f, W_f = features.shape
            # Reshape features to sequence: (B, HW, 512)
            feat_seq = features.view(B, C_f, -1).transpose(1, 2)
            
            # Prepare Query from Goal: (B, 1, 512)
            query = self.goal_proj(raw_goal).unsqueeze(1)
            
            # 1. Goal-driven spatial attention (Where is the goal target?)
            # Query: (B, 1, 512), Key/Value: (B, HW, 512)
            attn_out, _ = self.cross_attn(query, feat_seq, feat_seq)
            goal_context = attn_out.squeeze(1) # (B, 512)
            
            # 2. Global spatial pooling (What is the whole scene doing?)
            # Global Pooling -> (B, 512, 1, 1) -> (B, 512)
            global_context = self.pool(features).flatten(1)
            
            # 3. Combine them (addition preserves the 512D shape for existing checkpoints)
            pooled = global_context + goal_context
        else:
            # Fallback for inference without goals (if needed)
            # Global Pooling -> (B, 512, 1, 1) -> (B, 512)
            pooled = self.pool(features).flatten(1)
            
        # Project to Latent
        latent = self.proj(pooled) # (B, 256)
        
        if return_preds:
            # Decode for reconstruction
            # Reshape latent to (B, 256, 1, 1) for ConvTranspose
            latent_spatial = latent.view(latent.size(0), latent.size(1), 1, 1)
            reconstruction = self.decoder(latent_spatial)
            
            # Predict probes
            pred_proprio = self.proprio_head(latent)
            pred_end_pose = self.end_pose_head(latent)
            pred_obj_props = self.obj_props_head(latent)
            
            return latent, {
                'reconstruction': reconstruction,
                'proprio': pred_proprio,
                'end_pose': pred_end_pose,
                'object_props': pred_obj_props
            }
            
        return latent

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



