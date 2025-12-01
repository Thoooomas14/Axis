import sys
import os
import torch
import torch.nn as nn
import argparse
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.train import train
from training.data.rtx_loader import RTXDataset

def mock_dataset_iter():
    # Yield dummy batches: (images, proprio, action, goal_embs)
    # B=2
    while True:
        images = torch.randn(2, 3, 128, 128)
        proprio = torch.randn(2, 6) # 6D EE Pose
        action = torch.randn(2, 6) # 6D EE Pose Target
        goal_embs = torch.randn(2, 77) # Structured Goal
        yield images, proprio, action, goal_embs

def test_train_loop():
    print("Testing Training Loop...")
    
    # Mock RTXDataset
    # We monkeypatch the class to return our mock iterator
    original_iter = RTXDataset.__iter__
    RTXDataset.__iter__ = mock_dataset_iter
    
    # Mock __init__ to do nothing
    original_init = RTXDataset.__init__
    RTXDataset.__init__ = lambda self, **kwargs: None

    try:
        args = argparse.Namespace(
            dataset='dummy',
            batch_size=2,
            lr=1e-4,
            steps=5, # Run 5 steps
            warmup_steps=1,
            save_interval=5,
            checkpoint_dir='tests/checkpoints',
            embeddings_path='tests/dummy_embeddings.pkl',
            mock=True
        )
        
        # Run training
        train(args)
        print("Training loop ran successfully.")
        
        # Check if checkpoint was created
        if os.path.exists('tests/checkpoints/checkpoint_5.pt'):
            print("Checkpoint created.")
        else:
            print("Checkpoint NOT created.")
            sys.exit(1)

    except Exception as e:
        print(f"Training loop failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Restore mocks
        RTXDataset.__iter__ = original_iter
        RTXDataset.__init__ = original_init
        
        # Cleanup
        if os.path.exists('tests/checkpoints'):
            import shutil
            shutil.rmtree('tests/checkpoints')

if __name__ == "__main__":
    test_train_loop()
