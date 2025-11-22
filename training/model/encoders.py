import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Applies an affine transformation to the input feature map based on a conditioning embedding.
    """
    def __init__(self, num_features, cond_dim):
        super().__init__()
        self.num_features = num_features
        self.fc = nn.Linear(cond_dim, 2 * num_features)

    def forward(self, x, cond):
        """
        Args:
            x: Input feature map (B, C, H, W) or (B, C)
            cond: Conditioning embedding (B, cond_dim)
        """
        # Project condition to scale (gamma) and shift (beta)
        params = self.fc(cond)
        gamma, beta = torch.split(params, self.num_features, dim=1)

        # Reshape for broadcasting
        if x.dim() == 4:
            gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
            beta = beta.view(beta.size(0), beta.size(1), 1, 1)
        else:
            gamma = gamma.view(gamma.size(0), gamma.size(1))
            beta = beta.view(beta.size(0), beta.size(1))

        return (1 + gamma) * x + beta

class VisionEncoder(nn.Module):
    """
    Encodes RGB images into a feature map, conditioned on a goal embedding via FiLM.
    Uses a simplified ResNet-like structure for real-time performance.
    """
    def __init__(self, input_channels=3, base_channels=32, cond_dim=256):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, base_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 1
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels * 2)
        self.film1 = FiLMLayer(base_channels * 2, cond_dim)

        # Layer 2
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(base_channels * 4)
        self.film2 = FiLMLayer(base_channels * 4, cond_dim)

        # Layer 3
        self.conv4 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(base_channels * 8)
        self.film3 = FiLMLayer(base_channels * 8, cond_dim)
        
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, cond):
        """
        Args:
            x: Images (B, C, H, W)
            cond: Goal embedding (B, cond_dim)
        Returns:
            Feature map (B, C_out, H_out, W_out)
        """
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.film1(x, cond)
        x = self.act(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.film2(x, cond)
        x = self.act(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.film3(x, cond)
        x = self.act(x)

        return x

class ProprioEncoder(nn.Module):
    """
    Encodes robot proprioceptive state (End-Effector Pose) into a latent vector.
    Default input_dim=6 (3 Pos + 3 Rot).
    """
    def __init__(self, input_dim=6, output_dim=256, hidden_dim=128):
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
        return self.net(x)

class GoalEncoder(nn.Module):
    """
    Project semantic goal embeddings (from LLM/Gemini) into the model's dimension.
    """
    def __init__(self, input_dim=768, output_dim=256): # Default input_dim 768 for standard BERT/Gemini embeddings
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
