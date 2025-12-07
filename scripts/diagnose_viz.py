import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imitation.utils.visualizer import Visualizer

def diagnose():
    print("Starting Visualization Diagnostic...")
    
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Initializing Visualizer with save_dir={save_dir}...")
    try:
        viz = Visualizer(save_dir)
        print(f"Visualizer initialized. Internal save_dir: {viz.save_dir}")
    except Exception as e:
        print(f"FAIL: Could not initialize Visualizer: {e}")
        return

    # Test 1: Direct Matplotlib Save
    print("\nTest 1: Direct Matplotlib Save...")
    try:
        plt.switch_backend('Agg')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_title("Test Plot")
        
        test_path = os.path.join(viz.save_dir, 'test_direct.png')
        plt.savefig(test_path)
        plt.close()
        
        if os.path.exists(test_path):
            print(f"SUCCESS: Saved {test_path}")
        else:
            print(f"FAIL: File {test_path} was not created (no error raised).")
    except Exception as e:
        print(f"FAIL: Matplotlib error: {e}")

    # Test 2: Visualizer.visualize_batch
    print("\nTest 2: Visualizer.visualize_batch...")
    try:
        # Mock Batch
        B = 1
        batch = {
            'images': torch.randn(B, 8, 3, 128, 128),
            'action': torch.randn(B, 8) # 8D
        }
        pred_action = torch.randn(B, 8)
        
        viz.visualize_batch(0, batch, pred_action, save_prefix='diag_viz')
        
        expected_path = os.path.join(viz.save_dir, 'diag_viz_step_0.png')
        if os.path.exists(expected_path):
            print(f"SUCCESS: Saved {expected_path}")
        else:
            print(f"FAIL: File {expected_path} was not created.")
    except Exception as e:
        print(f"FAIL: visualize_batch error: {e}")

    print("\nDiagnostic Complete.")

if __name__ == "__main__":
    diagnose()
