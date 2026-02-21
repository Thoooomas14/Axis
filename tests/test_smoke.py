import pytest
import torch
import sys
import os

# Add project root to path ensuring we can import from training
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.axis import AxisModel

def test_model_dimensions():
    """Verifies that the AxisModel accepts inputs and produces outputs with expected shapes."""
    print("Verifying Axis Model Dimensions...")
    
    # Config matching SE(3) full matrix representation
    config = {
        'device': 'cpu',
        'proprio_dim': 13,  # SE(3) uses 13D: R_flat(9) + Trans(3) + Gripper(1)
        'action_dim': 7,    # Twist: 3 AngVel + 3 LinVel + 1 Gripper
        'goal_dim': 38,
        'embed_dim': 256,
        'vision_feature_dim': 256,
        'num_vision_tokens': 8,
        'num_heads': 4,
        'num_layers': 2 # Small for speed
    }
    
    model = AxisModel(config)
    model.eval()
    
    # Dummy Inputs
    B = 2
    W = 8 # Window size 8
    C, H, W_img = 3, 128, 128
    
    images = torch.randn(B, W, C, H, W_img)
    proprio = torch.randn(B, W, 13)  # 13D Proprio
    goal = torch.randn(B, 64)  # 64D Goal
    
    # Forward Pass
    action, requery = model(images, proprio, goal)
    
    # Assertions - Model outputs (B, W, action_dim) due to chunking
    assert action.shape == (B, W, 7), f"Expected Action (B, W, 7), got {action.shape}"
    assert requery.shape == (B, 1), f"Expected Requery (B, 1), got {requery.shape}"

def test_model_instantiation():
    """Simple test to check model instantiation with default config."""
    config = {
        'device': 'cpu', 
        'proprio_dim': 13,
        'action_dim': 7
    }
    model = AxisModel(config)
    assert isinstance(model, AxisModel)

