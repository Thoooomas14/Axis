import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imitation.data.rtx_loader import RTXDataset

def visualize_episode(args):
    print("Visualizing Training Episode...")
    
    # Load Dataset
    dataset = RTXDataset(
        dataset_name=args.dataset, 
        split='train', 
        batch_size=1, # Get 1 full episode
        image_size=(128, 128),
        data_dir=args.data_dir,
        shuffle=True # Random episode
    )
    
    # Get one episode
    iterator = iter(dataset)
    batch = next(iterator)
    # Unpack 6 items: images, proprio, action, goal_embs, mask, requery_labels
    images, proprio, action, goal_embs, mask = batch[0], batch[1], batch[2], batch[3], batch[4]
    
    # Shapes:
    # Images: (1, T, 3, H, W)
    # Proprio: (1, T, 7)
    # Action: (1, T, 7)
    # Mask: (1, T)
    
    T = images.shape[1]
    print(f"Loaded episode with {T} steps.")
    
    # Convert to numpy
    # Permute images to (T, H, W, C) for PIL/Matplotlib
    images_np = images[0].permute(0, 2, 3, 1).numpy()
    # Denormalize images if they were normalized (RTXDataset usually yields 0-1 floats)
    images_np = (images_np * 255).astype(np.uint8)
    
    proprio_np = proprio[0].numpy()
    action_np = action[0].numpy()
    mask_np = mask[0].numpy()
    
    # Filter masked steps (if any padding exists, though batch_size=1 usually yields full length up to max_len)
    valid_indices = np.where(mask_np > 0)[0]
    valid_T = len(valid_indices)
    print(f"Valid steps (unmasked): {valid_T}")
    
    images_np = images_np[valid_indices]
    proprio_np = proprio_np[valid_indices]
    action_np = action_np[valid_indices]
    
    # 1. Save GIF
    print("Saving Episode GIF...")
    frames = [Image.fromarray(img) for img in images_np]
    gif_path = 'episode_viz.gif'
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=100, # 100ms per frame = 10fps
        loop=0
    )
    print(f"Saved GIF to {gif_path}")
    
    # 2. Plot Trajectories
    print("Plotting Trajectories...")
    # 8 Dimensions: X, Y, Z, Qx, Qy, Qz, Qw, Gripper
    fig, axes = plt.subplots(4, 2, figsize=(15, 15))
    fig.suptitle(f"Episode Trajectories (Length {valid_T})")
    
    labels = ['X', 'Y', 'Z', 'Qx', 'Qy', 'Qz', 'Qw', 'Gripper']
    
    for i in range(8):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # Plot Proprio (Current State)
        ax.plot(proprio_np[:, i], label='Proprio (Current)', color='blue', linewidth=2)
        
        # Plot Action (Target State)
        ax.plot(action_np[:, i], label='Action (Target)', color='red', linestyle='--', alpha=0.7)
        
        ax.set_title(labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend()
        
    plt.tight_layout()
    plot_path = 'episode_trajectories.png'
    plt.savefig(plot_path)
    print(f"Saved Trajectories to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fractal20220817_data')
    parser.add_argument('--data_dir', type=str, default='data/content/axis_data/fractal20220817_data/0.1.0')
    
    args = parser.parse_args()
    visualize_episode(args)
