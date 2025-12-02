import torch
import numpy as np
import argparse
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

def analyze_data(args):
    print(f"Analyzing data in: {args.data_dir}")
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.pt")))
    print(f"Found {len(files)} episodes.")
    
    all_actions = []
    all_proprio = []
    all_requery = []
    episode_lengths = []
    gripper_states = []
    
    for f in tqdm(files):
        data = torch.load(f)
        
        # Unpack
        # We want the raw data before padding/truncation if possible, 
        # but the .pt files from preprocess_dataset.py contain the FULL sequence (variable length)
        # as I implemented it to save "data_dict" with tensors of length T.
        
        actions = data['action'].numpy() # (T, 6)
        proprio = data['proprio'].numpy() # (T, 6)
        requery = data['requery_labels'].numpy() # (T,)
        
        # Gripper is usually the last dim of proprio or action?
        # In preprocess: 
        # proprio = get_pose(step)[:6] -> This is EE pose (x,y,z, rx,ry,rz). No gripper.
        # action = world_vector + rotation_delta. No gripper?
        # Wait, where is the gripper action?
        # In RTXDataset/preprocess, I only extracted 6D action.
        # "action = tf.concat(parts, axis=-1)" where parts are world_vector (3) and rotation_delta (3).
        # The gripper action is usually separate in 'gripper_closedness_action' or similar.
        # I might have missed extracting the gripper action!
        
        # Let's check preprocess_dataset.py again.
        # Yes, I only extract 'world_vector' and 'rotation_delta'.
        # If the robot needs to control the gripper, we need that action dimension!
        
        all_actions.append(actions)
        all_proprio.append(proprio)
        all_requery.append(requery)
        episode_lengths.append(len(actions))
        
    # Concatenate
    all_actions = np.concatenate(all_actions, axis=0)
    all_proprio = np.concatenate(all_proprio, axis=0)
    all_requery = np.concatenate(all_requery, axis=0)
    
    print("\n--- Statistics ---")
    print(f"Total Steps: {len(all_actions)}")
    print(f"Avg Episode Length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")
    
    print("\nAction Stats (Global):")
    print(f"Mean: {np.mean(all_actions, axis=0)}")
    print(f"Std:  {np.std(all_actions, axis=0)}")
    print(f"Min:  {np.min(all_actions, axis=0)}")
    print(f"Max:  {np.max(all_actions, axis=0)}")
    
    print("\nRequery Stats:")
    print(f"Total Requeries: {np.sum(all_requery)}")
    print(f"Requery Rate: {np.mean(all_requery):.4f}")
    
    # Plotting
    os.makedirs(args.output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.hist(episode_lengths, bins=50)
    plt.title("Episode Length Distribution")
    plt.savefig(os.path.join(args.output_dir, "episode_lengths.png"))
    plt.close()
    
    # Action Histograms
    plt.figure(figsize=(15, 10))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.hist(all_actions[:, i], bins=50, alpha=0.7)
        plt.title(f"Action Dim {i}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "action_dist.png"))
    plt.close()
    
    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/processed/fractal_v1')
    parser.add_argument('--output_dir', type=str, default='analysis_results')
    args = parser.parse_args()
    analyze_data(args)
