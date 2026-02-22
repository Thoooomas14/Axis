import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.axis import AxisModel

def test_token_caching():
    print("Testing Token Caching Optimization...")
    
    # 1. Setup Model
    config = {
        'proprio_dim': 13,
        'action_dim': 7,
        'goal_dim': 38,
        'embed_dim': 512, # Vision(256) + Proprio(128) + Goal(128)
        'num_heads': 4,
        'num_layers': 2,
        'hidden_dim': 128,
        'window_size': 4
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = AxisModel(config).to(device)
    model.eval()
    
    # 2. Create Dummy Data for a sequence length of W+1
    # We will slide a window of size W over this.
    # Time steps: 0, 1, 2, 3, 4
    # Window 1: [0, 1, 2, 3]
    # Window 2: [1, 2, 3, 4]
    
    W = 4
    B = 1
    C, H = 3, 128
    
    # Sequence of 5 frames
    images_seq = torch.randn(B, W + 1, C, H, H).to(device)
    proprio_seq = torch.randn(B, W + 1, 13).to(device)
    goal = torch.randn(B, 38).to(device)
    
    # 3. Standard Execution (Ground Truth)
    # Run fully on Window 2
    win2_imgs = images_seq[:, 1:1+W]
    win2_props = proprio_seq[:, 1:1+W]
    
    print("\nRunning Standard Pass (Target)...")
    with torch.no_grad():
        out_act_std, out_req_std = model(win2_imgs, win2_props, goal)
        
    # 4. Cached Execution
    # First, run Window 1 to populate cache
    win1_imgs = images_seq[:, 0:W]
    win1_props = proprio_seq[:, 0:W]
    
    print("Running Window 1 (Prime Cache)...")
    with torch.no_grad():
        _, _, tokens_w1 = model(win1_imgs, win1_props, goal, return_tokens=True)
        
    # Prepare cache for Window 2:
    # Take last W-1 tokens from Window 1 result
    # tokens_w1 shape: (B, W, 512) -> we want indices 1..W-1 (which correspond to time steps 1..3)
    cached_tokens = tokens_w1[:, 1:, :] 
    
    print(f"Cached tokens shape: {cached_tokens.shape} (Expected: {B}, {W-1}, {512})")
    
    # Run Window 2 with cache
    # We pass the full window images/proprio, but the model should only use the last frame
    print("Running Cached Pass...")
    with torch.no_grad():
        out_act_cached, out_req_cached, _ = model(win2_imgs, win2_props, goal, 
                                               cached_tokens=cached_tokens, 
                                               return_tokens=True)
        
    # 5. Verify Results
    diff_act = torch.abs(out_act_std - out_act_cached).max().item()
    diff_req = torch.abs(out_req_std - out_req_cached).max().item()
    
    print(f"\nMax Action Difference: {diff_act}")
    print(f"Max Requery Difference: {diff_req}")
    
    TOLERANCE = 1e-4
    if diff_act < TOLERANCE and diff_req < TOLERANCE:
        print("\n✅ SUCCESS: Cached output matches standard output.")
    else:
        print("\n❌ FAILURE: Mismatch detected!")
        sys.exit(1)

if __name__ == "__main__":
    test_token_caching()
