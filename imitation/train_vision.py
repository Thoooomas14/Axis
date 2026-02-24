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
import pypose as pp

def se3_pose_loss(pred_13d: torch.Tensor, target_13d: torch.Tensor, omega_rot=1.0, omega_trans=1.0) -> torch.Tensor:
    """
    Computes SE(3) chordal loss between predicted and target 13D poses.
    Args:
        pred_13d: (B, 13) [R_flat(9), pos(3), gripper(1)]
        target_13d: (B, 13) [R_flat(9), pos(3), gripper(1)]
    """
    # 1. Rotation (Chordal distance: ||R_pred - R_target||_F^2)
    R_pred = pred_13d[:, :9].reshape(-1, 3, 3)
    R_targ = target_13d[:, :9].reshape(-1, 3, 3)
    rot_loss = torch.sum((R_pred - R_targ) ** 2, dim=(-2, -1)).mean()
    
    # 2. Translation (MSE)
    trans_loss = torch.sum((pred_13d[:, 9:12] - target_13d[:, 9:12]) ** 2, dim=-1).mean()
    
    # 3. Gripper (MSE)
    grip_loss = torch.mean((pred_13d[:, 12] - target_13d[:, 12]) ** 2)
    
    return omega_rot * rot_loss + omega_trans * trans_loss + grip_loss

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
            max_episodes=args.max_episodes,
            mix_episodes=args.mix_episodes,
            ram_usage_limit=args.ram_limit
        )
    else:
        log.info(f"Initializing Streaming DataLoader: {args.dataset}")
        stream_loader = RTXStreamLoader(
            data_dir=args.data_dir,
            dataset_name=args.dataset,
            split='train',
            window_size=args.batch_size, # Random frames use window_size as batch size to yield
            loss_horizon=1,
            use_subprocess=args.use_subprocess,
            shuffle_buffer_size=args.shuffle_buffer_size,
            mix_episodes=args.mix_episodes,
            ram_usage_limit=args.ram_limit
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
    model = VisionEncoder(feature_dim=256, goal_dim=38, pretrained=True).to(device)
    log.info("VisionEncoder initialized.")
    
    # --- Dynamic Loss Weighting (Uncertainty) ---
    # We initialize 4 log-variances for 4 tasks: [recon, proprio, end_pose, obj_props]
    log_vars = nn.Parameter(torch.zeros(4, device=device))
    
    # Add log_vars to the optimizer so they can be learned alongside the model!
    optimizer = optim.AdamW([
        {'params': model.parameters()},
        {'params': log_vars, 'lr': args.lr, 'weight_decay': 0.0} # No weight decay on log_vars
    ], lr=args.lr, weight_decay=1e-4)
    
    scaler = torch.cuda.amp.GradScaler()
    criterion_mse = nn.MSELoss()
    
    model.train()
    
    step = 0
    pbar = tqdm(total=args.steps, desc="Training Vision Encoder")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Running statistics for target normalization (to balance loss)
    # We normalize targets [Proprio, EndPose] independently to roughly zero-mean unit-variance
    target_mean_p = torch.zeros(13).to(device)
    target_std_p = torch.ones(13).to(device)
    target_mean_e = torch.zeros(13).to(device)
    target_std_e = torch.ones(13).to(device)
    alpha_ema = 0.01 
    
    for batch in dataloader:
        if step >= args.steps:
            break
            
        # Unpack Data
        images = batch['images'].to(device, non_blocking=True).squeeze(0) # (W, 3, 128, 128)
        proprio_raw = batch['proprio'].to(device, non_blocking=True).squeeze(0) # (W, 13)
        end_pose_raw = batch['subtask_end_pose'].to(device, non_blocking=True).squeeze(0) # (W, 13)
        object_props = batch['object_props'].to(device, non_blocking=True).squeeze(0) # (9,)
        
        # Expand episode-level object_props to match the window batch dimension
        object_props = object_props.unsqueeze(0).expand(images.size(0), -1) # (W, 9)
        
        # 1. Update/Apply Target Normalization
        with torch.no_grad():
            batch_mean_p = proprio_raw.mean(0)
            batch_std_p = proprio_raw.std(0).clamp(min=1e-3)
            target_mean_p = (1 - alpha_ema) * target_mean_p + alpha_ema * batch_mean_p
            target_std_p = (1 - alpha_ema) * target_std_p + alpha_ema * batch_std_p
            
            batch_mean_e = end_pose_raw.mean(0)
            batch_std_e = end_pose_raw.std(0).clamp(min=1e-3)
            target_mean_e = (1 - alpha_ema) * target_mean_e + alpha_ema * batch_mean_e
            target_std_e = (1 - alpha_ema) * target_std_e + alpha_ema * batch_std_e
            
        # NO NORMALIZATION FOR SE(3) CHORDAL TARGETS
        # Chordal loss mathematically requires raw SO(3) matrices and unscaled translations
        proprio = proprio_raw  
        end_pose = end_pose_raw 
            
        # The dataloader directly provides the encoded 38D 'goal' using GoalOracle
        raw_goal = batch['goal'].to(device, non_blocking=True).squeeze(0) # (W, 38)
        
        # --- NOISY GOAL DROPOUT (Prevent Information Leak) ---
        # Goal indices: 3-15 (Start Pose), 16-28 (Target Pose), 29-37 (Object properties)
        # We want the VisionEncoder to guess the exact properties and pose from the IMAGE,
        # so we degrade the `raw_goal` query to force it to look at the image features.
        noisy_goal = raw_goal.clone()
        
        # Add uniform noise to the target pose query (indices 16-28) to simulate a rough/flawed real-world goal
        # Position noise: +/- 10cm, Rotation noise: +/- small amount
        pose_noise = (torch.rand_like(noisy_goal[:, 16:29]) * 2 - 1) * 0.1 
        noisy_goal[:, 16:29] += pose_noise
        
        # Completely drop out (zero) the object properties 50% of the time
        # This forces the obj_props_head to literally learn what the object looks like from the image
        mask = (torch.rand(noisy_goal.size(0), 1, device=device) > 0.5).float()
        noisy_goal[:, 29:38] = noisy_goal[:, 29:38] * mask
        # -----------------------------------------------------
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed Precision Forward
        with torch.cuda.amp.autocast():
            # Process Vision using the NOISY GOAL
            latent, preds = model(images, raw_goal=noisy_goal, return_preds=True)
            
            # Losses are calculated against the PERFECT targets
            # 1. Reconstruction Loss
            recon_loss = criterion_mse(preds['reconstruction'], images)
            
            # 2. Proprioception Loss (Where am I?) - SE(3) Chordal
            proprio_loss = se3_pose_loss(preds['proprio'], proprio, omega_rot=1.0, omega_trans=1.0)
            
            # 3. Subtask End Pose Loss (Where is the exact target?) - SE(3) Chordal
            end_pose_loss = se3_pose_loss(preds['end_pose'], end_pose, omega_rot=1.0, omega_trans=1.0)
            
            # 4. Object Properties Loss (What EXACTLY is it?)
            obj_props_loss = criterion_mse(preds['object_props'], object_props)
            
            # Total Loss = sum( 1 / (2*sigma^2) * MSE + log(sigma) )
            # We use formulation exp(-log_var) for stability, and learn log_var directly.
            loss = 0.0
            
            w_recon = torch.exp(-log_vars[0]) * recon_loss + log_vars[0]
            w_proprio = torch.exp(-log_vars[1]) * proprio_loss + log_vars[1]
            w_end = torch.exp(-log_vars[2]) * end_pose_loss + log_vars[2]
            w_obj = torch.exp(-log_vars[3]) * obj_props_loss + log_vars[3]
            
            # Note: factor of 0.5 is mathematically standard but empirically often absorbed into lr,
            # so we just directly sum the scaled components for simplicity.
            loss = sum([w_recon, w_proprio, w_end, w_obj])
        
        # Scaled Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        # Clip max norm, including log_vars
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [log_vars], max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        step += 1
        pbar.update(1)
        # Calculate visible sigma = exp(0.5 * log_var) for easy monitoring
        with torch.no_grad():
            sigmas = torch.exp(0.5 * log_vars)
            
        # Log the components AND standard deviations to monitor balance
        pbar.set_postfix({
            'L': f"{loss.item():.2f}", 
            'Reconst': f"{w_recon.item():.2f} (s={sigmas[0].item():.2f})",
            'Proprio': f"{w_proprio.item():.2f} (s={sigmas[1].item():.2f})",
            'EndPose': f"{w_end.item():.2f} (s={sigmas[2].item():.2f})",
            'ObjProps': f"{w_obj.item():.2f} (s={sigmas[3].item():.2f})"
        })
        
        # Save Checkpoint, including the learned log_vars
        if step % args.save_interval == 0 or step == args.steps:
            save_path = os.path.join(args.save_dir, 'vision_encoder_latest.pt')
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'log_vars': log_vars.data,
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
    parser.add_argument('--ram_limit', type=float, default=12.0, help="RAM usage limit in GB for data workers")
    parser.add_argument('--mix_episodes', type=int, default=8, help="Number of episodes to mix in RAM")
    parser.add_argument('--shuffle_buffer_size', type=int, default=1000)
    parser.add_argument('--local_data_path', type=str, default=None, help="Path to local HDF5 file (overrides streaming)")
    parser.add_argument('--max_episodes', type=int, default=0, help="Max episodes to use (0=all)")
    
    args = parser.parse_args()
    train_vision(args)
