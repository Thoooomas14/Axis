import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.axis_v1 import AxisModel

def verify_model():
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
    
    print(f"Input Shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Proprio: {proprio.shape}")
    print(f"  Goal: {goal.shape}")
    
    # Forward Pass
    try:
        # use_memory=False by default now
        action, next_latent, requery = model(images, proprio, goal)
        
        print("\nOutput Shapes:")
        print(f"  Action: {action.shape}")
        print(f"  Next Latent: {next_latent.shape}")
        print(f"  Requery: {requery.shape}")
        
        # Assertions
        assert action.shape == (B, 8), f"Expected Action (B, 8), got {action.shape}"
        assert requery.shape == (B, 1), f"Expected Requery (B, 1), got {requery.shape}"
        
        print("\n✅ Verification Successful: Dimensions match requirements.")
        
    except Exception as e:
        print(f"\n❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_model()
