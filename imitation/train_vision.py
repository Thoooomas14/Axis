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
        
        # Optimization: Load larger chunks (windows) to reduce HDF5 I/O overhead
        CHUNK_WINDOW = 50 
        
        stream_loader = LocalDataLoader(
            data_path=args.local_data_path,
            window_size=CHUNK_WINDOW, 
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
        batch_size=args.batch_size, # Let torch collate
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
    
    # Weights for different explainable targets
    alpha_recon = 1.0 # Reconstruction
    alpha_proprio = 1.0 # Current State
    alpha_end = 1.0 # Subtask End State (Target)
    alpha_obj = 1.0 # Object Properties
    
    model.train()
    
    step = 0
    pbar = tqdm(total=args.steps, desc="Training Vision Encoder")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    running_loss = 0.0
    
    for batch in dataloader:
        if step >= args.steps:
            break
            
        # Unpack Data
        # Loader yields dicts where each value is (B, W, ...) or (B, ...).
        # Torch collates them into tensors.
        # With window_size=1, we expect images to be (B, 1, 3, 128, 128).
        
        # Data is yielded as (1, W, ...) where W = args.batch_size
        images = batch['images'].to(device, non_blocking=True).squeeze(0) # (W, 3, 128, 128)
        proprio = batch['proprio'].to(device, non_blocking=True).squeeze(0) # (W, 13)
        subtask_end_pose = batch['subtask_end_pose'].to(device, non_blocking=True).squeeze(0) # (W, 13)
        object_props = batch['object_props'].to(device, non_blocking=True).squeeze(0) # (W, 9)
        
        # Flattening not needed as we squeezed the B=1 dimension
        # and images is now (W, 3, 128, 128) which matches model input.
            
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed Precision Forward
        with torch.cuda.amp.autocast():
            latent, preds = model(images, return_preds=True)
            
            # 1. Reconstruction Loss
            recon_loss = criterion_mse(preds['reconstruction'], images)
            
            # 2. Proprioception Loss (Where am I?)
            proprio_loss = criterion_mse(preds['proprio'], proprio)
            
            # 3. Subtask End Pose Loss (Where is the target?)
            end_pose_loss = criterion_mse(preds['end_pose'], subtask_end_pose)
            
            # 4. Object Properties Loss (What is it?)
            obj_props_loss = criterion_mse(preds['object_props'], object_props)
            
            # Total Loss
            loss = (alpha_recon * recon_loss +
                    alpha_proprio * proprio_loss +
                    alpha_end * end_pose_loss +
                    alpha_obj * obj_props_loss)
        
        # Scaled Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        step += 1
        pbar.update(1)
        pbar.set_postfix({
            'L': f"{loss.item():.4f}", 
            'R': f"{recon_loss.item():.4f}",
            'P': f"{proprio_loss.item():.4f}",
            'E': f"{end_pose_loss.item():.4f}",
            'O': f"{obj_props_loss.item():.4f}"
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
