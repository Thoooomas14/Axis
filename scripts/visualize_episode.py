import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imitation.data.rtx_stream_loader import RTXStreamLoader
from imitation.data.local_loader import LocalDataLoader
from imitation.utils.visualizer import Visualizer

def visualize_episode(args):
    print("Visualizing Training Episode...")
    
    # --- Load Dataset ---
    if args.local_path:
        print(f"Loading Local Dataset from {args.local_path}")
        dataset = LocalDataLoader(
            data_path=args.local_path,
            batch_size=1,
            window_size=args.window_size,
            loss_horizon=1,
            repeat=False # One epoch
        )
    else:
        print(f"Loading Streaming Dataset {args.dataset}")
        dataset = RTXStreamLoader(
            dataset_name=args.dataset, 
            split='train', 
            batch_size=1, # Get 1 full episode
            window_size=args.window_size,
            image_size=(128, 128),
            data_dir=args.data_dir,
            repeat=False, # One epoch
            use_subprocess=True
        )
    
    # Get one episode (stream mimics windows, so we just take a batch of windows or a full episode if iter returns full - WAIT)
    # Both Loaders return WINDOWS now, not full episodes directly in one batch unless we modify them.
    # RTXStreamLoader yields DICTS with keys 'images' (B, W, ...).
    # To visualize a full episode, we need to collect windows or use a special mode.
    # Actually, for visualization, seeing a sequence of windows is fine, or we can try to stitch.
    
    # Let's take just ONE window batch for detailed inspection using the new Visualizer
    iterator = iter(dataset)
    try:
        batch = next(iterator)
    except StopIteration:
        print("Dataset is empty.")
        return

    # Unpack
    # Batch is a DICT now: {'images', 'proprio', 'goal', 'actions', 'target_poses'}
    print("Keys in batch:", batch.keys())
    
    # Create Visualizer
    viz = Visualizer(save_dir='.')
    
    # Fake a prediction (just use GT + noise for demo)
    gt_action = batch['actions'] # (B, H, 7)
    # We visualizing the LAST step of the window
    # gt_action is (B, H, 7), we want (B, 7) for single step
    tgt_act = gt_action[:, 0, :] # First horizon step
    
    # Fake Pred
    pred_act = tgt_act + torch.randn_like(tgt_act) * 0.1
    
    print("Visualizing Batch Window...")
    viz.visualize_batch(step=0, batch=batch, pred_action=pred_act, save_prefix='viz_debug')
    
    print(f"Saved visualization to ./visualizations/viz_debug_step_0.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fractal20220817_data')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--local_path', type=str, default=None, help="Path to .h5 file. If provided, uses LocalDataLoader.")
    parser.add_argument('--window_size', type=int, default=8)
    
    args = parser.parse_args()
    visualize_episode(args)
