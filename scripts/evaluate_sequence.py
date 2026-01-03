import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.axis_v1 import AxisModel
from imitation.data.rtx_loader import RTXDataset

def evaluate_sequence(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Configuration ---
    config = {
        'device': device,
        'goal_dim': 64,
        'cond_dim': 256,
        'vision_feature_dim': 256,
        'num_vision_tokens': 8,
        'window_size': 8,
        'latent_dim': 256,
        'queue_size': 10,
        'embed_dim': 256,
        'num_heads': 4,
        'num_layers': 4,
        'proprio_dim': 8,
        'action_dim': 8
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
    
    # Filter out latent_queue state if shapes don't match (e.g. different batch size)
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
    # --- Load Data ---
    print("Loading data...")
    
    if args.use_processed_data:
        from imitation.data.processed_loader import ProcessedDataset
        from torch.utils.data import DataLoader
        
        dataset = ProcessedDataset(args.data_dir, max_len=64)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        iterator = iter(loader)
        batch = next(iterator)
        # Unpack
        images, proprio, target_action, goal_embs, mask, requery_labels = batch
    else:
        # Note: RTXDataset now yields full episodes (B, T, ...)
        dataset = RTXDataset(
            dataset_name=args.dataset, 
            split='train', 
            batch_size=1, # Get 1 episode
            image_size=(128, 128),
            data_dir=args.data_dir,
            shuffle=False
        )

        # Get one batch (one episode)
        iterator = iter(dataset)
        # Unpack 6 items: images, proprio, action, goal_embs, mask, requery_labels
        batch = next(iterator)
        images, proprio, target_action, goal_embs, mask = batch[0], batch[1], batch[2], batch[3], batch[4]

    # Move to device
    # Shape: (B, T, ...)
    images = images.to(device)
    proprio = proprio.to(device)
    goal_embs = goal_embs.to(device)
    mask = mask.to(device)

    B, T = images.shape[0], images.shape[1]
    print(f"Loaded episode with shape: {images.shape}")

    # --- Inference Loop ---
    pred_actions = []
    
    # Reset memory
    model.reset_memory(B)
    
    W_size = config['window_size']
    
    with torch.no_grad():
        for t in range(T):
            # Slice window
            if t < W_size - 1:
                pad_len = W_size - 1 - t
                img_slice = images[:, :t+1]
                prop_slice = proprio[:, :t+1]
                img_pad = img_slice[:, 0:1].repeat(1, pad_len, 1, 1, 1)
                prop_pad = prop_slice[:, 0:1].repeat(1, pad_len, 1)
                img_window = torch.cat([img_pad, img_slice], dim=1)
                prop_window = torch.cat([prop_pad, prop_slice], dim=1)
            else:
                img_window = images[:, t-W_size+1 : t+1]
                prop_window = proprio[:, t-W_size+1 : t+1]

            # Forward
            pred_action, _, _ = model(
                img_window, 
                prop_window, 
                goal_embs[:, t], # Pass goal for current step (B, 77)
                robot_name='default', 
                update_queue=True, # Update memory during inference
                use_memory=True
            )
            pred_actions.append(pred_action.cpu())

    pred_actions = torch.stack(pred_actions, dim=1) # (B, T, 7)

    # --- Visualization ---
    print("Generating visualization...")
    
    # Take first element in batch
    images_np = images[0].cpu().permute(0, 2, 3, 1).numpy()
    target_action_np = target_action[0].cpu().numpy()
    pred_action_np = pred_actions[0].numpy()
    mask_np = mask[0].cpu().numpy()
    
    # Limit to first 20 steps or T
    viz_steps = min(T, 20)
    
    # Create figure
    # Create figure
    fig, axes = plt.subplots(viz_steps, 2, figsize=(12, 4 * viz_steps))
    action_labels = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'g']

    for i in range(viz_steps):
        # 1. Image
        ax_img = axes[i, 0]
        ax_img.imshow(images_np[i])
        ax_img.set_title(f"Step {i} (Mask: {mask_np[i]})")
        ax_img.axis('off')
        
        # 2. Action Comparison
        ax_act = axes[i, 1]
        x = np.arange(8)
        width = 0.35
        
        ax_act.bar(x - width/2, target_action_np[i], width, label='Ground Truth', color='green', alpha=0.7)
        ax_act.bar(x + width/2, pred_action_np[i], width, label='Prediction', color='red', alpha=0.7)
        
        ax_act.set_ylabel('Value')
        ax_act.set_title('Action Prediction')
        ax_act.set_xticks(x)
        ax_act.set_xticklabels(action_labels)
        if i == 0: ax_act.legend()
        ax_act.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = 'evaluation_sequence.png'
    plt.savefig(output_path)
    print(f"Saved sequential visualization to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fractal20220817_data')
    parser.add_argument('--data_dir', type=str, default='data/content/axis_data/fractal20220817_data/0.1.0')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--use_processed_data', action='store_true', help='Use preprocessed .pt files instead of raw TFDS')
    
    args = parser.parse_args()
    evaluate_sequence(args)
