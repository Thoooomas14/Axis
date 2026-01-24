import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.axis_v1 import AxisModel
from imitation.data.rtx_stream_loader import RTXStreamLoader
from imitation.data.local_loader import LocalDataLoader
from imitation.utils.visualizer import Visualizer

def evaluate_sequence(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Configuration ---
    config = {
        'device': device,
        'goal_dim': 38,
        'cond_dim': 256,
        'vision_feature_dim': 256,
        'num_vision_tokens': 8,
        'window_size': 8,
        'latent_dim': 256,
        'queue_size': 10,
        'embed_dim': 256,
        'num_heads': 4,
        'num_layers': 4,
        'proprio_dim': 13, # [R(9), p(3), g(1)]
        'action_dim': 7,    # [v(3), w(3), g(1)]
        'use_action_chunking': True,
        'chunk_size': 1 # Default
    }

    # --- Load Model ---
    model = AxisModel(config).to(device)
    model.eval()

    if not os.path.exists(args.checkpoint_dir):
        print(f"Checkpoint directory {args.checkpoint_dir} not found.")
        return

    checkpoints = [f for f in os.listdir(args.checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        print("No checkpoints found.")
        return

    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    latest_checkpoint = checkpoints[-1]
    checkpoint_path = os.path.join(args.checkpoint_dir, latest_checkpoint)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Filter out latent_queue state if shapes don't match
    state_dict = checkpoint['model_state_dict']
    model_state = model.state_dict()
    
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state:
            if v.shape != model_state[k].shape:
                print(f"Skipping {k} due to shape mismatch: {v.shape} vs {model_state[k].shape}")
                continue
            filtered_state_dict[k] = v
            
    model.load_state_dict(filtered_state_dict, strict=False)

    # --- Load Data ---
    print("Loading data...")
    if args.local_path:
         print(f"Loading local data from {args.local_path}")
         dataset = LocalDataLoader(
             data_path=args.local_path,
             batch_size=1,
             window_size=8,
             repeat=False
         )
    else:
        print(f"Loading streaming data {args.dataset}")
        dataset = RTXStreamLoader(
            dataset_name=args.dataset, 
            split='train', 
            batch_size=1, 
            window_size=8,
            image_size=(128, 128),
            data_dir=args.data_dir,
            repeat=False,
            use_subprocess=False # In-process for simpler debugging
        )

    # Get one batch
    iterator = iter(dataset)
    try:
        batch = next(iterator)
    except StopIteration:
        print("Dataset empty.")
        return
        
    # Unpack batch (torch tensors)
    images = batch['images'].to(device)    # (B, W, 3, H, W)
    proprio = batch['proprio'].to(device)  # (B, W, 13)
    goal = batch['goal'].to(device)        # (B, 38)
    gt_action = batch['actions'].to(device) # (B, H, 7)
    
    B, W = images.shape[0], images.shape[1]
    print(f"Loaded batch: Img {images.shape}, Prop {proprio.shape}, Goal {goal.shape}")
    
    # --- Prediction ---
    model.reset_memory(B)
    
    with torch.no_grad():
        # Predict for the last step in window
        # In sliding window training, we usually predict actions for the *last* frame
        # using the history.
        
        # Forward pass
        # model expects (B, W, ...)
        output = model(
            images, 
            proprio, 
            goal.unsqueeze(1).repeat(1, W, 1), # (B, W, 38) - goal is static per window
            robot_name='default',
            update_queue=True,
            use_memory=True
        )
        
        # Output is often a dict in newer AxisModel? Need to verify return type.
        # Assuming tuple (pred_action, ..., ...) based on previous file
        if isinstance(output, tuple):
             pred_action_chunk, _, _ = output
        elif isinstance(output, dict):
             pred_action_chunk = output['action_chunks']
        else:
             pred_action_chunk = output
             
        # pred_action_chunk: (B, ChunkSize, 7)
        # We take the first step of the chunk
        pred_action = pred_action_chunk[:, 0, :] # (B, 7)

    # --- Visualization ---
    print("Generating visualization...")
    viz = Visualizer(save_dir='.')
    viz.visualize_batch(step=0, batch=batch, pred_action=pred_action, save_prefix='eval_test')
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fractal20220817_data')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--local_path', type=str, default=None, help="Path to local .h5 file. Uses LocalLoader if set.")
    
    args = parser.parse_args()
    evaluate_sequence(args)
