import os
import sys
import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.encoders import VisionEncoder
from imitation.data.local_loader import LocalDataLoader
from imitation.data.rtx_stream_loader import RTXStreamLoader

def evaluate_vision(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    model = VisionEncoder(feature_dim=256, goal_dim=38, pretrained=False).to(device)
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])    
        step = checkpoint.get('step', 'Unknown')
        print(f"Loaded checkpoint from {args.checkpoint} (Step: {step})")
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}. Using untrained model weights.")

    model.eval()

    # Load Data
    if args.local_data_path:
        print(f"Loading sample data from local HDF5: {args.local_data_path}")
        stream_loader = LocalDataLoader(
            data_path=args.local_data_path,
            window_size=args.batch_size,
            loss_horizon=1,
            shuffle=True,
            repeat=False,
            max_episodes=1 # Just need one episode to grab a few frames
        )
    else:
        print(f"Loading sample data from streaming dataset: {args.dataset}")
        stream_loader = RTXStreamLoader(
            data_dir=None, # Uses default GS bucket for DROID
            dataset_name=args.dataset,
            split='train',
            batch_size=8, # workers batch size 
            window_size=args.batch_size, # This dictates the number of frames we evaluate
            loss_horizon=1,
            use_subprocess=args.use_subprocess,
            shuffle_buffer_size=10
        )

    # Note: StreamLoader yields dicts matching local loader.
    dataloader = torch.utils.data.DataLoader(stream_loader, batch_size=1, num_workers=0)

    try:
        batch = next(iter(dataloader))
    except StopIteration:
        print("Failed to load any data from the provided path.")
        return

    # Unpack and prep data
    images = batch['images'].to(device).squeeze(0) # (W, 3, 128, 128)
    raw_goal = batch['goal'].to(device).squeeze(0) # (W, 38)
    proprio_raw = batch['proprio'].to(device).squeeze(0) # (W, 13)
    end_pose_raw = batch['subtask_end_pose'].to(device).squeeze(0) # (W, 13)
    object_props = batch['object_props'].to(device).squeeze(0) # (9,)
    object_props = object_props.unsqueeze(0).expand(images.size(0), -1) # (W, 9)

    # Normalize targets using independent batch statistics for evaluation purposes
    batch_mean_p = proprio_raw.mean(0)
    batch_std_p = proprio_raw.std(0, unbiased=False).clamp(min=1e-3)
    proprio = (proprio_raw - batch_mean_p) / batch_std_p
    
    batch_mean_e = end_pose_raw.mean(0)
    batch_std_e = end_pose_raw.std(0, unbiased=False).clamp(min=1e-3)
    end_pose = (end_pose_raw - batch_mean_e) / batch_std_e

    # Forward Pass
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            print("Running forward pass...")
            # We don't apply noisy dropout during evaluation, pass the clean goal
            latent, preds = model(images, raw_goal=raw_goal, return_preds=True)
            
            # --- DEBUG ---
            print(f"Latent shape: {latent.shape}")
            print(f"Latent Mean (per channel limit): min={latent.mean(0).min().item():.4f}, max={latent.mean(0).max().item():.4f}")
            print(f"Latent Std (per channel): {latent.std(0).mean().item():.4f} (Average over channels)")
            print(f"Latent Std (across batch): {latent.std(dim=0).mean().item():.4f}")
            print(f"Image Min: {images.min().item():.4f}, Max: {images.max().item():.4f}")
            print(f"Recon Min: {preds['reconstruction'].min().item():.4f}, Max: {preds['reconstruction'].max().item():.4f}")
            # -------------

    # Calculate MSE vs Ground Truth
    criterion_mse = torch.nn.MSELoss()
    recon_mse = criterion_mse(preds['reconstruction'], images).item()
    proprio_mse = criterion_mse(preds['proprio'], proprio).item()
    end_pose_mse = criterion_mse(preds['end_pose'], end_pose).item()
    obj_props_mse = criterion_mse(preds['object_props'], object_props).item()

    print("\n--- Feature Evaluation vs Ground Truth (MSE) ---")
    mse_text = (
        f"Reconstruction MSE:  {recon_mse:.4f}\n"
        f"Proprioception MSE:  {proprio_mse:.4f}\n"
        f"End Pose MSE:        {end_pose_mse:.4f}\n"
        f"Object Props MSE:    {obj_props_mse:.4f}"
    )
    print(mse_text)
    print("------------------------------------------------\n")

    # Move to CPU for plotting
    imgs_np = images.cpu().numpy()
    recons_np = preds['reconstruction'].float().cpu().numpy()

    # Plot up to 8 frames
    num_to_plot = min(8, imgs_np.shape[0])
    
    # Increase height slightly to fit text at bottom
    fig, axes = plt.subplots(2, num_to_plot, figsize=(3 * num_to_plot, 7), squeeze=False)
    fig.suptitle(f"Vision Encoder Evaluation", fontsize=16)

    for i in range(num_to_plot):
        # Original Image
        orig_img = np.transpose(imgs_np[i], (1, 2, 0))
        orig_img = np.clip(orig_img, 0, 1) # Images are 0-1 floats
        axes[0, i].imshow(orig_img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title("Original Ground Truth")
        
        # Reconstructed Image
        recon_img = np.transpose(recons_np[i], (1, 2, 0))
        recon_img = np.clip(recon_img, 0, 1)
        axes[1, i].imshow(recon_img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title("Reconstruction")

    # Leave room at the bottom for the text block
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    
    # Add the MSE text at the bottom center
    fig.text(0.5, 0.02, mse_text, ha='center', va='bottom', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.savefig(args.output)
    print(f"Saved evaluation visualization to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/vision/vision_encoder_latest.pt', help='Path to model checkpoint')
    parser.add_argument('--local_data_path', type=str, default=None, help='Path to hdf5 dataset (overrides streaming)')
    parser.add_argument('--dataset', type=str, default='droid', help="Streaming dataset name (e.g. droid, fractal20220817_data)")
    parser.add_argument('--use_subprocess', action='store_true', help="Use subprocess for TF data loading (recommended)")
    parser.add_argument('--batch_size', type=int, default=8, help='Number of frames to visualize')
    parser.add_argument('--output', type=str, default='vision_eval.png', help='Output image filename')
    
    args = parser.parse_args()
    evaluate_vision(args)
