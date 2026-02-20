import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
import time
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.encoders import VisionEncoder
from imitation.data.rtx_stream_loader import RTXStreamLoader
from imitation.data.local_loader import LocalDataLoader

# Setup Logging
def setup_logging(verbose=True):
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)

log = logging.getLogger(__name__)

def train_vision(args):
    global log
    log = setup_logging()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")
    
    # Enable TF32 for faster float32 math
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- Data Loader ---
    # --- Data Loader ---
    if args.local_data_path:
        log.info(f"Initializing LocalDataLoader: {args.local_data_path}")
        # Note: LocalDataLoader yields windows. 
        # We need individual frames or just treat windows as batch?
        # Vision pre-training treats every frame as independent.
        # Window size 1 effectively gives us independent frames (mostly).
        
        # Optimization: Use window_size = batch_size to minimize HDF5 reads.
        # This makes exactly ONE read per training step for the entire batch.
        stream_loader = LocalDataLoader(
            data_path=args.local_data_path,
            window_size=args.batch_size, 
            loss_horizon=1,
            shuffle=True,
            repeat=True,
            max_episodes=args.max_episodes
        )
    else:
        log.info(f"Initializing Streaming DataLoader: {args.dataset}")
        stream_loader = RTXStreamLoader(
            data_dir=args.data_dir,
            dataset_name=args.dataset,
            split='train',
            batch_size=args.batch_size,
            window_size=1,
            loss_horizon=1,
            use_subprocess=args.use_subprocess,
            shuffle_buffer_size=args.shuffle_buffer_size
        )
    
    dataloader = torch.utils.data.DataLoader(
        stream_loader,
        batch_size=1, # One batch = one window of size args.batch_size
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    # --- Model ---
    model = VisionEncoder(feature_dim=256, pretrained=True).to(device)
    log.info("VisionEncoder initialized.")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    # --- Losses ---
    criterion_mse = nn.MSELoss()
    
    # Weights for different explainable targets (Balanced for magnitude)
    alpha_recon = 1000.0 # Reconstruction is small MSE
    alpha_proprio = 0.1  # Poses have large magnitudes
    alpha_end = 0.1
    alpha_obj = 1000.0
    
    model.train()
    
    step = 0
    pbar = tqdm(total=args.steps, desc="Training Vision Encoder")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Running statistics for target normalization (to balance loss)
    # We normalize targets [Proprio, EndPose] to roughly zero-mean unit-variance
    target_mean = torch.zeros(13).to(device)
    target_std = torch.ones(13).to(device)
    alpha_ema = 0.01 
    
    for batch in dataloader:
        if step >= args.steps:
            break
            
        # Unpack Data
        images = batch['images'].to(device, non_blocking=True).squeeze(0) # (W, 3, 128, 128)
        proprio_raw = batch['proprio'].to(device, non_blocking=True).squeeze(0) # (W, 13)
        end_pose_raw = batch['subtask_end_pose'].to(device, non_blocking=True).squeeze(0) # (W, 13)
        object_props = batch['object_props'].to(device, non_blocking=True).squeeze(0) # (W, 9)
        
        # 1. Update/Apply Target Normalization
        with torch.no_grad():
            batch_mean = proprio_raw.mean(0)
            batch_std = proprio_raw.std(0).clamp(min=1e-3)
            target_mean = (1 - alpha_ema) * target_mean + alpha_ema * batch_mean
            target_std = (1 - alpha_ema) * target_std + alpha_ema * batch_std
            
        # Normalize targets
        proprio = (proprio_raw - target_mean) / target_std
        end_pose = (end_pose_raw - target_mean) / target_std
            
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed Precision Forward
        with torch.cuda.amp.autocast():
            latent, preds = model(images, return_preds=True)
            
            # 1. Reconstruction Loss
            recon_loss = criterion_mse(preds['reconstruction'], images)
            
            # 2. Proprioception Loss (Where am I?)
            proprio_loss = criterion_mse(preds['proprio'], proprio)
            
            # 3. Subtask End Pose Loss (Where is the target?)
            end_pose_loss = criterion_mse(preds['end_pose'], end_pose)
            
            # 4. Object Properties Loss (What is it?)
            obj_props_loss = criterion_mse(preds['object_props'], object_props)
            
            # Weighted components
            w_recon = alpha_recon * recon_loss
            w_proprio = alpha_proprio * proprio_loss
            w_end = alpha_end * end_pose_loss
            w_obj = alpha_obj * obj_props_loss
            
            # Total Loss
            loss = w_recon + w_proprio + w_end + w_obj
        
        # Scaled Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        step += 1
        pbar.update(1)
        # Log the WEIGHTED values to monitor balance
        pbar.set_postfix({
            'L': f"{loss.item():.2f}", 
            'wR': f"{w_recon.item():.2f}",
            'wP': f"{w_proprio.item():.2f}",
            'wE': f"{w_end.item():.2f}",
            'wO': f"{w_obj.item():.2f}"
        })
        
        # Save Checkpoint
        if step % args.save_interval == 0 or step == args.steps:
            save_path = os.path.join(args.save_dir, 'vision_encoder_latest.pt')
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            }, save_path)
            # log.info(f"Saved checkpoint to {save_path}")

    pbar.close()
    log.info("Vision Pre-training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None, help="Path to dataset")
    parser.add_argument('--dataset', type=str, default='fractal20220817_data', help="Dataset name")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_vision', help="Directory to save checkpoints")
    parser.add_argument('--use_subprocess', action='store_true', help="Use subprocess for data loading")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of data loader workers")
    parser.add_argument('--shuffle_buffer_size', type=int, default=1000)
    parser.add_argument('--local_data_path', type=str, default=None, help="Path to local HDF5 file (overrides streaming)")
    parser.add_argument('--max_episodes', type=int, default=0, help="Max episodes to use (0=all)")
    
    args = parser.parse_args()
    train_vision(args)
