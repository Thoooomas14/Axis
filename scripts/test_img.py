import sys
import os

import torch

import numpy as np

from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.encoders import VisionEncoder

if __name__ == "__main__":
    path1 = "scripts/exterior_left_224_100.png"
    img1 = Image.open(path1).convert("RGB")
    img_data1 = np.array(img1)
    print("Image data shape:", img_data1.shape)  # (224, 224, 3)
    # Create a random input tensor simulating a batch of 4 RGB images of size 224x224
    img_data1 = img_data1.astype(np.float32).transpose(2, 0, 1)  # (3, 224, 224)
    img_data1 = torch.from_numpy(img_data1).unsqueeze(0).float()  # [B, C, H, W]

    path2 = "scripts/exterior_left_224_0.png"
    img2 = Image.open(path2).convert("RGB")
    img_data2 = np.array(img2)
    print("Image data shape:", img_data2.shape)  # (224, 224, 3)
    # Create a random input tensor simulating a batch of 4 RGB images of size 224x224
    img_data2 = img_data2.astype(np.float32).transpose(2, 0, 1)  # (3, 224, 224)
    img_data2 = torch.from_numpy(img_data2).unsqueeze(0).float()  # [B, C, H, W]

    # Instantiate the visual encoder
    encoder = VisionEncoder(feature_dim=768)

    # Forward pass through the encoder
    output1 = encoder(img_data1)
    output2 = encoder(img_data2)

    print("Output shape:", output1.shape)
    print("Output shape:", output2.shape)
