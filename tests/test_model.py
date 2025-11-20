import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.model.axis_v1 import AxisModel

def test_axis_model_forward():
    print("Testing AxisModel Forward Pass...")
    
    # Configuration
    config = {
        'device': 'cpu',
        'goal_dim': 768,
        'cond_dim': 256,
        'vision_feature_dim': 256,
        'num_vision_tokens': 8,
        'latent_dim': 256,
        'queue_size': 5,
        'embed_dim': 256,
        'num_heads': 4,
        'num_layers': 2,
        # Multi-robot config
        'robots': {
            'ur3e': {'proprio_dim': 7, 'action_dim': 7},
            'widowx': {'proprio_dim': 8, 'action_dim': 8}
        }
    }

    model = AxisModel(config)
    model.eval()

    # Dummy Inputs
    B = 2
    images = torch.randn(B, 3, 128, 128) # (B, C, H, W)
    goal = torch.randn(B, 768)           # (B, GoalDim)

    # --- Test Robot 1: UR3e ---
    print("Testing Robot: UR3e")
    proprio_ur3e = torch.randn(B, 7)
    
    # Check Latent Queue Init
    print(f"Initial Queue Shape: {model.latent_queue.queue.shape}")

    # Forward Pass 1
    print("Running Forward Pass 1 (UR3e)...")
    action, next_latent = model(images, proprio_ur3e, goal, robot_name='ur3e', update_queue=True)
    
    print(f"Action Shape: {action.shape}")
    assert action.shape == (B, 7)
    assert next_latent.shape == (B, 256)

    # --- Test Robot 2: WidowX ---
    print("Testing Robot: WidowX")
    proprio_widowx = torch.randn(B, 8)
    
    # Forward Pass 2
    print("Running Forward Pass 2 (WidowX)...")
    action2, next_latent2 = model(images, proprio_widowx, goal, robot_name='widowx', update_queue=True)
    
    print(f"Action Shape: {action2.shape}")
    assert action2.shape == (B, 8)
    
    print("Test Passed!")

if __name__ == "__main__":
    test_axis_model_forward()
