import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.model.axis_v1 import AxisModel
from training.data.rtx_loader import RTXDataset
from training.utils.gemini_mock import MockGeminiPlanner

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Configuration (Must match training) ---
    config = {
        'device': device,
        'goal_dim': 77,
        'cond_dim': 256,
        'vision_feature_dim': 256,
        'num_vision_tokens': 8,
        'latent_dim': 256,
        'queue_size': 10,
        'embed_dim': 256,
        'num_heads': 4,
        'num_layers': 4,
        'robots': {
            'default': {'proprio_dim': 6, 'action_dim': 6},
        }
    }

    # --- Load Model ---
    model = AxisModel(config).to(device)
    model.eval()

    # Find latest checkpoint
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
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded step {checkpoint['step']}")

    # --- Load Data ---
    # We use a small batch size for visualization
    dataset = RTXDataset(
        dataset_name=args.dataset, 
        split='train', 
        batch_size=args.num_samples,
        image_size=(128, 128),
        data_dir=args.data_dir
    )

    # Get one batch
    iterator = iter(dataset)
    images, proprio, target_action, goal_embs = next(iterator)

    # Move to device
    images_dev = images.to(device)
    proprio_dev = proprio.to(device)
    goal_embs_dev = goal_embs.to(device)

    # --- Inference ---
    with torch.no_grad():
        pred_action, _, _ = model(
            images_dev, 
            proprio_dev, 
            goal_embs_dev, 
            robot_name='default', 
            update_queue=False 
        )

    # --- Visualization ---
    print("Generating visualization...")
    
    # Move to CPU for plotting
    images_np = images.permute(0, 2, 3, 1).numpy() # N, H, W, C
    target_action_np = target_action.numpy()
    pred_action_np = pred_action.cpu().numpy()

    # Print Statistics
    print("\n--- Action Statistics ---")
    print(f"Target: Min={target_action_np.min():.4f}, Max={target_action_np.max():.4f}, Mean={target_action_np.mean():.4f}, Std={target_action_np.std():.4f}")
    print(f"Pred  : Min={pred_action_np.min():.4f}, Max={pred_action_np.max():.4f}, Mean={pred_action_np.mean():.4f}, Std={pred_action_np.std():.4f}")
    print("-------------------------\n")
    
    # Create figure
    num_samples = args.num_samples
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    action_labels = ['x', 'y', 'z', 'rx', 'ry', 'rz']

    for i in range(num_samples):
        # 1. Image
        ax_img = axes[i, 0]
        ax_img.imshow(images_np[i])
        ax_img.set_title(f"Sample {i}")
        ax_img.axis('off')
        
        # 2. Action Comparison (Bar Chart)
        ax_act = axes[i, 1]
        x = np.arange(6)
        width = 0.35
        
        ax_act.bar(x - width/2, target_action_np[i], width, label='Ground Truth', color='green', alpha=0.7)
        ax_act.bar(x + width/2, pred_action_np[i], width, label='Prediction', color='red', alpha=0.7)
        
        ax_act.set_ylabel('Value')
        ax_act.set_title('Action Prediction (Delta)')
        ax_act.set_xticks(x)
        ax_act.set_xticklabels(action_labels)
        ax_act.legend()
        ax_act.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = 'evaluation_viz.png'
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fractal20220817_data')
    parser.add_argument('--data_dir', type=str, default='data/content/axis_data/fractal20220817_data/0.1.0')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--num_samples', type=int, default=4)
    
    args = parser.parse_args()
    evaluate(args)
