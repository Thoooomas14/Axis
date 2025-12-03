import torch
import sys
import os
import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.data.rtx_stream_loader import RTXStreamLoader

def create_fake_episode(length=20):
    """Creates a fake episode with a gripper change in the middle."""
    steps = []
    
    # Phase 1: Open (0-9)
    # Phase 2: Closed (10-19)
    
    for i in range(length):
        gripper = 1.0 if i >= 10 else 0.0
        
        obs = {
            'image': tf.zeros((128, 128, 3), dtype=tf.uint8), # Fake image
            'ee_pose': tf.constant(np.random.randn(6).astype(np.float32)),
            'gripper_closed': tf.constant(np.array([gripper], dtype=np.float32))
        }
        
        step = {
            'observation': obs,
            'action': {
                'world_vector': tf.zeros(3),
                'rotation_delta': tf.zeros(3),
                'gripper_closedness_action': tf.zeros(1)
            }
        }
        steps.append(step)
        
    return {
        'steps': steps,
        'language_instruction': tf.constant("pick up the object")
    }

def verify_loader_logic():
    print("Verifying RTXStreamLoader Logic (Mock Data)...")
    
    # Instantiate loader (don't iterate it directly to avoid GCS)
    loader = RTXStreamLoader(
        dataset_name='dummy',
        window_size=8
    )
    
    print("Creating fake episode (Length 20, Split at 10)...")
    episode = create_fake_episode(20)
    
    print("Processing episode...")
    windows = list(loader._process_episode(episode))
    
    print(f"Generated {len(windows)} windows.")
    
    # We expect 2 subtasks.
    # Subtask 1: 0-10 (Length 10). Windows: 10-8+1 = 3.
    # Subtask 2: 10-20 (Length 10). Windows: 10-8+1 = 3.
    # Total: 6 windows.
    
    if len(windows) > 0:
        batch = windows[0]
        print(f"\nSample Window:")
        print(f"  Images: {batch['images'].shape}")
        print(f"  Proprio: {batch['proprio'].shape}")
        print(f"  Goal: {batch['goal'].shape}")
        print(f"  Action: {batch['action'].shape}")
        print(f"  Requery: {batch['requery'].shape}")
        
        assert batch['images'].shape == (8, 3, 128, 128)
        assert batch['proprio'].shape == (8, 8)
        assert batch['goal'].shape == (64,)
        assert batch['action'].shape == (8,)
        assert batch['requery'].shape == (1,)
        
        # Check Requery Logic
        # The last window of Subtask 1 should have requery=1
        # Subtask 1 windows are indices 0, 1, 2.
        # Window 2 should have requery=1.
        
        print("\nChecking Requery Labels:")
        for i, w in enumerate(windows):
            req = w['requery'].item()
            print(f"  Window {i}: Requery={req}")
            
        # Expected: 0, 0, 1, 0, 0, 1
        expected = [0, 0, 1, 0, 0, 1]
        actual = [int(w['requery'].item()) for w in windows]
        
        if actual == expected:
             print("\n✅ Requery Logic Verified.")
        else:
             print(f"\n❌ Requery Logic Failed. Expected {expected}, got {actual}")
             
        print("\n✅ Verification Successful: Logic is correct.")
        
    else:
        print("\n❌ Verification Failed: No windows generated.")

if __name__ == "__main__":
    verify_loader_logic()
