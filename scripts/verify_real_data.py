import os
import sys
import argparse
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.train import train

def verify_real_data():
    print("Starting End-to-End Verification with Real Data...")
    
    # Arguments for a minimal run
    args = argparse.Namespace(
        dataset='fractal20220817_data', # Use downloaded dataset
        batch_size=2,
        lr=1e-4,
        steps=5, # Run only 5 steps
        warmup_steps=1,
        save_interval=5,
        checkpoint_dir='verification_checkpoints',
        embeddings_path='instruction_embeddings.pkl', # Will be ignored/mocked by new logic
        mock=False # Use REAL data
    )
    
    # Ensure data directory exists
    # The zip file extracted to data/content/axis_data
    # tfds.builder_from_directory requires the path to the specific version folder
    data_dir = os.path.abspath('data/content/axis_data/fractal20220817_data/0.1.0')
    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading/Loading data to: {data_dir}")
    
    args.data_dir = data_dir
    
    try:
        train(args)
        print("\nSUCCESS: Training loop completed with real data!")
    except Exception as e:
        print(f"\nFAILURE: Verification failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_real_data()
