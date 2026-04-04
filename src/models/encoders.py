import warnings
import torch
import torch.nn as nn

# Suppress unfixable 3rd-party library warnings from DINOv2 and UniDepth
warnings.filterwarnings("ignore", message=".*xFormers is not available.*")
warnings.filterwarnings("ignore", message=".*Importing from timm.models.layers is deprecated.*")
warnings.filterwarnings("ignore", message=".*To run evaluation you need KNN.*")
warnings.filterwarnings("ignore", message=".*User provided device_type of 'cuda', but CUDA is not available.*")
warnings.filterwarnings("ignore", message=".*weights_only.*")
warnings.filterwarnings("ignore", message=".*self\\.resolution_level not set.*")




class VisionEncoder(nn.Module):
    def __init__(
        self, img_size=(224, 224), feature_dim=768, pretrained=True, goal_dim=128
    ) -> None:
        super().__init__()

        if img_size[0] != img_size[1]:
            raise ValueError(
                f"VisionEncoder currently only supports square images. Got {img_size}. "
                "Please resize your input images to be square."
            )
        if img_size[0] % 16 != 0:
            raise ValueError(
                f"VisionEncoder currently requires input image dimensions to be divisible by 16. Got {img_size}. "
                "Please resize your input images accordingly."
            )
        self.image_size = img_size

        # Load Backbones
        self.dinov2 = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14_reg", pretrained=pretrained, trust_repo=True
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="torch.hub")
        self.unidepthv2 = torch.hub.load(
            repo_or_dir="lpiccinelli-eth/UniDepth",
            model="UniDepth",
            version="v2",
            backbone="vits14",
            pretrained=pretrained,
            trust_repo=True
        )

        setattr(self.unidepthv2, "resolution_level", 0.0)  # noqa: B010

        # 384 (DINO) + 1 (unidepth) = 385
        self.fusion_block = nn.Sequential(
            nn.Conv2d(385, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Downsample spatially to 4x4 to match the 1024 input of self.latent
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.latent = nn.Sequential(
            nn.Linear(1024 + goal_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, feature_dim),
            nn.ReLU(),
        )

        kernel = img_size[0] // 16
        self.depth_pool = nn.AvgPool2d(
            kernel_size=kernel, stride=kernel
        )  # convert 224x224 depth image into 16x16 feature map

    def forward(self, x, goal: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        if (H, W) != self.image_size:
            raise ValueError(
                f"Expected input image size {self.image_size}, but got {(H, W)}. "
                "Please resize the input images accordingly."
            )

        with torch.no_grad():
            dino_out = self.dinov2.forward_features(x)  # type: ignore #Returns [B, 261, 384] (1 CLS + 4 Reg + 256 Patches)
            dino_feats = (
                dino_out["x_norm_patchtokens"].permute(0, 2, 1).reshape(B, 384, 16, 16)
            )  # Turn into 384 chanels and 16x16 spatial map

            uni_out = self.unidepthv2.infer(x)  # type: ignore
            depth = uni_out["depth"]  # [B, 1, H, W]
            depth_feats = self.depth_pool(depth)  # [B, 1, 16, 16]

        combined = torch.cat([dino_feats, depth_feats], dim=1)

        fused = self.fusion_block(combined)  # [B, 64, 4, 4]
        fused_flat = torch.flatten(fused, 1)  # [B, 1024]

        fused_flat = torch.cat(
            [fused_flat, goal], dim=-1
        )  # Simple fusion of goal info by adding mean goal embedding to each spatial location

        return self.latent(fused_flat)  # [B, feature_dim]


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
            self.register_buffer("running_mean", torch.zeros(input_dim))
            self.register_buffer("running_std", torch.ones(input_dim))
            self.register_buffer(
                "count", torch.tensor(0, dtype=torch.long).unsqueeze(0)
            )
            self.ema_alpha = 0.1

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
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
                (1 - self.ema_alpha) * self.running_std
                + self.ema_alpha * batch_std.clamp(min=1e-6)
            )

        self.count += 1

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
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
