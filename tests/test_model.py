import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.model.axis_v1 import AxisModel

def test_axis_model_forward():
    print("Testing AxisModel Forward Pass...")
    
    device = 'cpu'
    
    # Configuration
    config = {
        'device': device,
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
            'default': {'proprio_dim': 6, 'action_dim': 6},
            # 'widowx': {'proprio_dim': 6, 'action_dim': 6}
        }
    }

    model = AxisModel(config)
    model.eval()

    # Dummy Inputs
    B = 2
    # Test 1: Default Robot (UR3e/Generic)
    print("Testing Robot: Default (6D EE Pose)")
    robot_name = 'default' 
    
    # Initial State
    # Input images: (B, C, H, W)
    images = torch.randn(B, 3, 128, 128)
    # Proprio: (B, 6) - EE Pose
    proprio = torch.randn(B, 6)
    # Goal: (B, 768)
    goal = torch.randn(B, 768)

    print(f"Initial Queue Shape: {model.latent_queue.queue.shape}")

    # Forward Pass
    print(f"Running Forward Pass ({robot_name})...")
    action, next_latent, requery_logit = model(images, proprio, goal, robot_name=robot_name)
    
    print(f"Action Shape: {action.shape}")
    print(f"Requery Logit Shape: {requery_logit.shape}")
    
    # Verify Shapes
    # Action should be (B, 6)
    assert action.shape == (B, 6), f"Expected action shape ({B}, 6), got {action.shape}"
    # Latent should be (B, 256)
    assert next_latent.shape == (B, 256), f"Expected latent shape ({B}, 256), got {next_latent.shape}"
    # Requery should be (B, 1)
    assert requery_logit.shape == (B, 1), f"Expected requery shape ({B}, 1), got {requery_logit.shape}"
    
    print("Test Passed!")

if __name__ == "__main__":
    test_axis_model_forward()
