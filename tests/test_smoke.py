import pytest
import torch
import sys
import os

# Add project root to path ensuring we can import from training
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.axis_v1 import AxisModel

def test_model_dimensions():
    """Verifies that the AxisModel accepts inputs and produces outputs with expected shapes."""
    print("Verifying Axis Model Dimensions...")
    
    # Config matching user requirements
    config = {
        'device': 'cpu',
        'proprio_dim': 8,
        'action_dim': 8,
        'goal_dim': 64,
        'embed_dim': 256,
        'vision_feature_dim': 256,
        'num_vision_tokens': 8,
        'latent_dim': 256,
        'queue_size': 10,
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
    proprio = torch.randn(B, W, 8) # 8D EE Pose
    goal = torch.randn(B, 64) # 64D Goal
    
    # Forward Pass
    # use_memory=False by default now
    action, next_latent, requery = model(images, proprio, goal)
    
    # Assertions
    assert action.shape == (B, 8), f"Expected Action (B, 8), got {action.shape}"
    assert requery.shape == (B, 1), f"Expected Requery (B, 1), got {requery.shape}"
    assert next_latent.shape == (B, config['latent_dim']), f"Expected Next Latent (B, {config['latent_dim']}), got {next_latent.shape}"

def test_model_instantiation():
    """Simple test to check model instantiation with default config."""
    config = {
        'device': 'cpu',
        'proprio_dim': 8,
        'action_dim': 8,
        'goal_dim': 64,
        'embed_dim': 256,
        'vision_feature_dim': 256,
        'num_vision_tokens': 8,
        'latent_dim': 256,
        'queue_size': 10,
        'num_heads': 4,
        'num_layers': 2
    }
    model = AxisModel(config)
    assert isinstance(model, AxisModel)
