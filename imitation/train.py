import os
import sys
import gc
import logging
import platform

# Suppress TF INFO/WARNING logs to reduce noise (e.g. OUT_OF_RANGE)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import pickle
from tqdm import tqdm
import shutil
from datetime import datetime
import pypose as pp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.axis import AxisModel
from imitation.data.rtx_stream_loader import RTXStreamLoader
from imitation.data.local_loader import LocalDataLoader
from imitation.utils.scheduler import CosineAnnealingWarmupRestarts
from imitation.utils.logger import TrainingLogger
from src.utils.rotation_utils import chordal_se3_loss, pose_13d_to_se3_matrix, apply_twist
from imitation.utils.visualizer import Visualizer
from imitation.utils.ema import EMA
import tensorflow_datasets as tfds
import tensorflow as tf
from collections import deque
import time
# Force TensorFlow to use CPU only (prevents VRAM fighting with PyTorch and CUDA errors in workers)
tf.config.set_visible_devices([], 'GPU')

import numpy as np
import torchvision.transforms as T

# --- Logging Setup ---
def setup_logging(verbose: int = 1):
    """Configure logging with verbosity levels. 0=WARNING, 1=INFO, 2=DEBUG."""
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(verbose, logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)



log = logging.getLogger(__name__)
CHUNK_SIZE = 10

class ThroughputMonitor:
    """
    Tracks wall-clock throughput using a sliding window.
    Accounts for I/O pauses better than standard EMA.
    """
    def __init__(self, window_size=100, total_steps=None):
        self.window = deque(maxlen=window_size)
        self.total_steps = total_steps
        self.start_time = time.time()
        self.window.append((self.start_time, 0)) # time, step_count
        
    def update(self, current_step):
        now = time.time()
        self.window.append((now, current_step))
        
    def get_stats(self):
        if len(self.window) < 2:
            return 0.0, "N/A"
            
        t_start, s_start = self.window[0]
        t_end, s_end = self.window[-1]
        
        duration = t_end - t_start
        steps = s_end - s_start
        
        if duration < 1e-6:
            return 0.0, "N/A"
            
        rate = steps / duration
        
        eta_str = "N/A"
        if self.total_steps and rate > 0:
            remaining = self.total_steps - s_end
            if self.total_steps == float('inf'): # Infinite epochs
                 eta_str = "??" 
            else:
                remaining_sec = remaining / rate
                eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining_sec))
            
        return rate, eta_str

def train(args):
    # Setup logging based on verbosity
    global log
    log = setup_logging(args.verbose)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")
    log.debug("Script started. Initializing...")

    config = {
        'device': device,
        'goal_dim': 38, 
        'cond_dim': 256,
        'window_size': args.window_size,
        'embed_dim': 256,
        'num_heads': 4,
        'num_layers': 4,
        'proprio_dim': 13, # R_flat(9) + Trans(3) + Gripper(1) (full SE(3) matrix + gripper)
        'action_dim': 7,   # 3 AngularVel + 3 LinearVel + 1 Gripper (twist + gripper)
        'chunk_size': CHUNK_SIZE,
    }
    
    # --- Endpoint Chordal Loss (task-oriented, trig-free) ---
    def orthonormalize_rotation(R):
        """
        Orthonormalize rotation matrices using SVD.
        Project onto the closest valid rotation matrix in Frobenius norm.
        R: (B, 3, 3)
        Returns: (B, 3, 3) valid rotation matrices
        """
        U, S, V = torch.svd(R)
        # Ensure det(U @ V.T) is 1 (handle reflection case)
        # Construct correction matrix
        with torch.no_grad():
             det = torch.det(U @ V.transpose(-2, -1))
             diag = torch.ones_like(S)
             diag[:, -1] = det
             
             # Reconstruct: R = U @ diag @ V.T
             R_new = U @ torch.diag_embed(diag) @ V.transpose(-2, -1)
        return R_new

    def endpoint_chordal_loss(pred_twists, start_poses, target_poses, horizon, omega_rot=1.0, omega_trans=1.0):
        """
        Compute SE(3) chordal loss on endpoint after twist rollout.
        
        Uses Frobenius norm on rotation matrix difference - NO trigonometric functions!
        
        Args:
            pred_twists: (B, W, 7) predicted twists [ω, v, gripper_delta] per step
            start_poses: (B, 13) starting pose [R_flat, pos, gripper] - last frame of input window
            target_poses: (B, 13) target pose at step t+horizon
            horizon: Number of twist steps to apply (1 <= horizon <= W)
            omega_rot: Weight for rotational component
            omega_trans: Weight for translational component
            
        Returns:
            Scalar loss: chordal distance on SE(3) + gripper loss
        """
        B = pred_twists.shape[0]
        device = pred_twists.device
        dtype = pred_twists.dtype
        
        # 1. Build Predicted SE(3) Transform by rollout from 13D pose
        # Extract R and p from 13D: [R_flat(9), pos(3), gripper(1)]
        R_start = start_poses[:, :9].reshape(B, 3, 3)  # (B, 3, 3)
        p_start = start_poses[:, 9:12]  # (B, 3)
        
        T_pred = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, 4, 4).clone()
        
        # Sanitize input rotation
        R_start_clean = orthonormalize_rotation(R_start)
        T_pred[:, :3, :3] = R_start_clean
        T_pred[:, :3, 3] = p_start
        
        # Initialize LieTensor state once
        # check=False is acceptable here because we just sanitized it
        T_pred_pp = pp.mat2SE3(T_pred, check=False) 
        
        gripper_curr = start_poses[:, 12:13]  # (B, 1)
        
        # Rollout: Apply predicted twists sequentially using PyPose
        for t in range(horizon):
            twist_6d = pred_twists[:, t, :6]  # (B, 6) - [ω, v]
            gripper_delta = pred_twists[:, t, 6:7]  # (B, 1)
            
            # Use PyPose for twist application
            T_delta_pp = pp.Exp(pp.se3(twist_6d))  # SE(3) delta
            
            # Update state in Lie Group (No matrix conversion inside loop)
            T_pred_pp = T_pred_pp @ T_delta_pp  # Body-frame composition
            
            gripper_curr = gripper_curr + gripper_delta
            
        # Final conversion to matrix for loss calculation
        T_pred = T_pred_pp.matrix()
        
        # 2. Build Target SE(3) Transform from 13D pose
        R_target = target_poses[:, :9].reshape(B, 3, 3)  # (B, 3, 3)
        p_target = target_poses[:, 9:12]  # (B, 3)
        
        T_target = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, 4, 4).clone()
        T_target[:, :3, :3] = R_target
        T_target[:, :3, 3] = p_target
        
        # 3. Compute Chordal Loss (Frobenius norm - NO TRIG!)
        R_pred = T_pred[:, :3, :3]
        p_pred = T_pred[:, :3, 3]
        
        # Rotation loss: ||R_pred - R_target||^2_F
        rot_diff = R_pred - R_target
        rot_loss = torch.mean(torch.sum(rot_diff ** 2, dim=(-2, -1)))
        
        # Translation loss: ||p_pred - p_target||^2
        trans_diff = p_pred - p_target
        trans_loss = torch.mean(torch.sum(trans_diff ** 2, dim=-1))
        
        # 4. Gripper Loss (separate, not part of SE(3))
        gripper_pred = torch.clamp(gripper_curr, 0.0, 1.0)
        gripper_target = target_poses[:, 12:13]
        grip_loss = torch.mean((gripper_pred - gripper_target)**2)
        
        return omega_rot * rot_loss + omega_trans * trans_loss + grip_loss

    # --- Utilities ---
    training_logger = TrainingLogger(args.checkpoint_dir)
    visualizer = Visualizer(args.checkpoint_dir)

    # --- Model ---
    model = AxisModel(config).to(device)
    log.info("Model initialized.")

    # Dataset
    # If epochs > 0, we do NOT repeat the dataset (finite epoch).
    # If steps > 0 (and epochs=0), we repeat the dataset (infinite stream).
    repeat_dataset = (args.epochs == 0)
    
    # === Dynamic Memory Parameters (can be reduced on OOM) ===
    effective_batch_size = args.batch_size
    effective_shuffle_buffer = args.shuffle_buffer_size
    effective_num_workers = args.num_workers
    
    def create_dataloader(batch_size, shuffle_buffer, num_workers, split='train', split_start=0.0, split_end=1.0):
        """Factory function to create/recreate dataloader with specified params."""
        
        # Use local loader if local_data_path is provided
        if args.local_data_path:
            log.info(f"Creating LOCAL dataloader: batch_size={batch_size}, split=[{split_start:.0%}-{split_end:.0%}]")
            stream = LocalDataLoader(
                data_path=args.local_data_path,
                window_size=config['window_size'],
                loss_horizon=args.loss_horizon,
                shuffle=('train' in str(split_start) or split_start == 0.0),
                repeat=(args.epochs == 0),
                split_start=split_start,
                split_end=split_end,
            )
            # Local loader supports multi-worker
            effective_workers = num_workers
        else:
            log.info(f"Creating dataloader ({split}): batch_size={batch_size}, use_subprocess={args.use_subprocess}")
            
            # RTXStreamLoader handles subprocess isolation internally when use_subprocess=True
            stream = RTXStreamLoader(
                data_dir=args.data_dir,
                dataset_name=args.dataset,
                split=split,
                image_key=args.image_key,
                image_size=(128, 128),
                window_size=config['window_size'],
                loss_horizon=args.loss_horizon,
                shuffle_buffer_size=shuffle_buffer,
                use_subprocess=args.use_subprocess,
                queue_size=64,
            )
            # When use_subprocess=True, the loader handles parallelism internally
            effective_workers = 0 if args.use_subprocess else num_workers
        
        loader = torch.utils.data.DataLoader(
            stream, 
            batch_size=batch_size,
            num_workers=effective_workers,
            pin_memory=True, # Enable pinning for speed
            prefetch_factor=4 if effective_workers > 0 else None,
            persistent_workers=True if effective_workers > 0 else False,
        )
        return loader, stream
    
    # Create initial dataloader
    # For streaming: use TFDS split syntax; for local: use split_start/split_end
    train_split = f'train[:{int(args.train_split_pct*100)}%]'
    val_split = f'train[{int(args.train_split_pct*100)}%:]'
    
    dataloader, train_stream = create_dataloader(
        effective_batch_size, effective_shuffle_buffer, effective_num_workers, 
        split=train_split, split_start=0.0, split_end=args.train_split_pct
    )
    if args.dataset == 'droid':
        avg_episode_length = 250
    else:
        avg_episode_length = 100
    # Estimate total windows for progress tracking (if method exists)
    if hasattr(train_stream, 'estimate_total_windows'):
        estimated_windows = train_stream.estimate_total_windows() if args.local_data_path else train_stream.estimate_total_windows(avg_episode_length=avg_episode_length)
        estimated_batches = estimated_windows // effective_batch_size if estimated_windows else None
        if estimated_batches:
            log.info(f"Estimated ~{estimated_windows:,} windows ({estimated_batches:,} batches) per epoch")
    else:
        estimated_batches = None
        log.info("Progress estimation not available for subprocess loader")
    
    # Validation Dataloader (smaller batch size to save memory if needed)
    if args.val_batch_size == 0:
        args.val_batch_size = effective_batch_size
    val_dataloader, _ = create_dataloader(
        args.val_batch_size, 0, 0, 
        split=val_split, split_start=args.train_split_pct, split_end=1.0
    )
    
    need_dataloader_rebuild = False  # Flag to trigger rebuild from outer loop
    
    # --- Optimizer & Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer, 
        first_cycle_steps=args.steps, 
        max_lr=args.lr, 
        min_lr=1e-6, 
        warmup_steps=args.warmup_steps
    )
    
    # --- GradScaler (AMP) ---
    scaler = torch.cuda.amp.GradScaler()
    
    # --- EMA ---
    ema = EMA(model, decay=0.9999)
    log.info("EMA initialized.")
    
    # --- Augmentation ---
    # Applied on GPU
    augmentations = nn.Sequential(
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.05),
        # Add slight blur occasionally
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.1)
    ).to(device)
    
    # Requery Loss Function (Option C: MSELoss with soft labels)
    # Using MSE allows the model to output confidence values [0, 1] directly
    # while avoiding the double-sigmoid issue of BCEWithLogitsLoss + Sigmoid layer
    requery_criterion = nn.MSELoss()

    # --- Training Loop ---
    model.train()
    
    # --- Resume from Checkpoint ---
    start_step = 0
    start_epoch = 0
    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pt')
    
    if args.resume and os.path.exists(checkpoint_path):
        log.info(f"Resuming from latest checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Filter state dict
        state_dict = checkpoint['model_state_dict']
        model_state = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
                
        model.load_state_dict(filtered_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        log.info(f"Resumed at step {start_step}, epoch {start_epoch}")
    elif os.path.exists(checkpoint_path):
        log.warning(f"Checkpoint found at {checkpoint_path} but --resume was not provided.")
        log.warning("Starting from scratch. Use --resume to continue training.")
    else:
        log.info("No checkpoint found. Starting from scratch.")

    # Determine total steps/epochs
    if args.epochs > 0:
        total_steps = args.steps # Fallback
        log.info(f"Training for {args.epochs} epochs.")
    else:
        total_steps = args.steps
        log.info(f"Training for {args.steps} steps.")

    # Calculate target step (relative to start)
    if args.epochs == 0:
        target_step = start_step + args.steps
    else:
        target_step = float('inf')
    
    log.info("Starting training...")
    
    import time
    import psutil  # For system RAM monitoring
    start_time = time.time()
    step = start_step
    
    # === Dynamic Memory Management State ===
    consecutive_cuda_oom = 0
    consecutive_ram_oom = 0
    max_consecutive_ooms = 3
    successful_batches_since_oom = 0
    
    # Minimum values before graceful exit
    MIN_BATCH_SIZE = 1
    MIN_SHUFFLE_BUFFER = 1
    MIN_NUM_WORKERS = 0 
    
    try:
        log.debug("Entering training loop...")
        
        # Loop structure
        num_epochs = args.epochs if args.epochs > 0 else 1
        
        # For step-based training, ignore epoch from checkpoint
        # (prevents range(51, 1) = empty when resuming epoch-trained checkpoint)
        if args.epochs == 0:
            start_epoch = 0
        
        # Global pbar for step-based training
        # Global pbar for step-based training
        if args.epochs == 0:
            pbar = tqdm(total=target_step, initial=start_step, desc="Training Steps", smoothing=0.0) # Disable tqdm smoothing
            monitor = ThroughputMonitor(window_size=100, total_steps=target_step) 
            monitor.update(start_step)
        
        steps_per_epoch_actual = None
        
        for epoch in range(start_epoch, num_epochs):
            steps_in_current_epoch = 0  # Initialize for both modes
            
            if args.epochs > 0:
                log.info(f"--- Epoch {epoch + 1}/{num_epochs} ---")
                
                # Dynamic Progress Bar with estimation
                # Use estimated_batches for first epoch, then actual count from previous epochs
                # Use estimated_batches if we haven't completed an epoch yet (e.g. start or resume)
                if steps_per_epoch_actual is None:
                    epoch_total = estimated_batches  # Use estimate for ETA (may be None)
                else:
                    epoch_total = steps_per_epoch_actual  # Actual count from previous epoch
                
                pbar = tqdm(total=epoch_total, desc=f"Epoch {epoch + 1}", smoothing=0.0)
                monitor = ThroughputMonitor(window_size=100, total_steps=epoch_total)
            
            for batch in dataloader:
                if args.epochs == 0 and step >= target_step:
                    break
                
                # Check time limit
                if args.time_limit_min > 0:
                    elapsed_min = (time.time() - start_time) / 60.0
                    if elapsed_min >= args.time_limit_min:
                        log.info(f"Time limit of {args.time_limit_min} minutes reached. Stopping.")
                        break
                
                # === OOM Protection: Wrap batch processing in try/except ===
                try:
                    # Periodic memory cleanup (every 1000 steps)
                    if step % 1000 == 0:
                        gc.collect()
                        # Only clear CUDA cache on OOM or very rarely to avoid sync overhead
                        # if torch.cuda.is_available():
                        #     torch.cuda.empty_cache()
                    
                    # Unpack Batch (requery is NOT in data - computed from action loss)
                    images = batch['images'].to(device)   # (B, W, 3, 128, 128)
                    proprio = batch['proprio'].to(device) # (B, W, 7)
                    goal_embs = batch['goal'].to(device)  # (B, 64)
                    target_actions = batch['actions'].to(device) # (B, H, 7) - future twists (H=loss_horizon)
                    target_poses = batch['target_poses'].to(device)  # (B, H, 7) - direct future poses

                    B = images.shape[0]
                    W = images.shape[1]  # Window size

                    # === Data Augmentation ===
                    if model.training:
                        # Reshape to (B*T, C, H, W) for efficient augmentation
                        fl_imgs = images.view(-1, 3, 128, 128)
                        aug_imgs = augmentations(fl_imgs)
                        images = aug_imgs.view(B, W, 3, 128, 128)

                    optimizer.zero_grad()
                    
                    # === Forward Pass with AMP ===
                    with torch.cuda.amp.autocast():
                        # Forward Pass - model outputs chunked predictions
                        pred_action, requery_pred = model(
                            images, 
                            proprio, 
                            goal_embs
                        )
                        
                        # pred_action: (B, W, 7) - chunked twist predictions
                        # requery_pred: (B, 1) - model confidence [0, 1]
                        
                        # === Endpoint Loss (Task-Oriented) ===
                        # Disable AMP for geodesic loss - SE(3) operations need float32 precision
                        with torch.cuda.amp.autocast(enabled=False):
                            # Cast to float32 for numerical precision in rotation ops
                            pred_action_f32 = pred_action.float()
                            start_poses_f32 = proprio[:, -1, :].float()  # (B, 13)
                            
                            # Target endpoint: directly from data loader at the horizon step
                            horizon = args.loss_horizon
                            target_endpoint_f32 = target_poses[:, horizon - 1, :].float()  # (B, 13)
                            
                            
                            # Compute endpoint loss in float32 (chordal - trig-free!)
                            action_loss = endpoint_chordal_loss(
                                pred_action_f32, start_poses_f32, target_endpoint_f32, 
                                horizon, args.omega_rot, args.omega_trans
                            )
                        
                        # Compute per-sample loss for confidence (using endpoint error)
                        per_sample_loss = action_loss.detach()  # Scalar, broadcast to all samples
                        
                        # === Confidence Target (Self-Supervised) ===
                        # Requery represents model confidence: high when predictions are accurate
                        confidence_target = torch.exp(-per_sample_loss / args.confidence_temperature).expand(B, 1)
                        confidence_target = confidence_target.detach()
                        
                        # Compute requery loss (MSE between predicted confidence and loss-derived target)
                        requery_loss = requery_criterion(requery_pred, confidence_target)
                        
                        # Weighted Loss
                        loss = action_loss + (requery_loss * args.requery_weight)
                    
                    # === Backward with Scaler ===
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # === EMA Update ===
                    ema.update(model)
                    
                    scheduler.step()
                    
                    # Logging
                    training_logger.log_step(step, epoch, loss.item(), action_loss.item(), requery_loss.item())
                    
                    step += 1
                    steps_in_current_epoch += 1
                    
                    # Update Monitor
                    monitor.update(step if args.epochs == 0 else steps_in_current_epoch)
                    rate, eta = monitor.get_stats()
                    
                    pbar.set_description(f"L:{loss.item():.4f} A:{action_loss.item():.4f} R:{requery_loss.item():.4f} | {rate:.2f}it/s | ETA: {eta}")
                    pbar.update(1)  # Increment progress bar counter
                    
                    # === Persistent Memory Fix ===
                    # Python doesn't always release memory to OS. We force it periodically.
                    if step % 1000 == 0:
                        gc.collect()

                    
                    # === Validation Loop ===
                    if step % args.val_interval == 0:
                        # Force GC to clear any transient training memory
                        gc.collect()
                        if torch.cuda.is_available(): os.environ.get('empty_cache', torch.cuda.empty_cache)()

                        model.eval()
                        val_loss_total = 0.0
                        val_batches = 0
                        log.info("Running Validation...")
                        with torch.no_grad():
                            for val_batch in val_dataloader:
                                if val_batches >= args.val_batches: break # Limit val batches
                                
                                v_imgs = val_batch['images'].to(device)
                                v_props = val_batch['proprio'].to(device)
                                v_goals = val_batch['goal'].to(device)
                                v_target_poses = val_batch['target_poses'].to(device)
                                
                                B_val = v_imgs.shape[0]
                                
                                with torch.cuda.amp.autocast():
                                    # Validation forward pass
                                    v_pred, v_requery_pred = model(v_imgs, v_props, v_goals)
                                    
                                # Endpoint loss in float32 (SE(3) ops need precision)
                                with torch.cuda.amp.autocast(enabled=False):
                                    v_pred_f32 = v_pred.float()
                                    v_start_poses_f32 = v_props[:, -1, :].float()
                                    horizon = args.loss_horizon
                                    v_target_endpoint_f32 = v_target_poses[:, horizon - 1, :].float()
                                    
                                    v_a_loss = endpoint_chordal_loss(
                                        v_pred_f32, v_start_poses_f32, v_target_endpoint_f32,
                                        horizon, args.omega_rot, args.omega_trans
                                    )
                                    
                                    v_confidence_target = torch.exp(-v_a_loss.detach() / args.confidence_temperature).expand(B_val, 1)
                                    v_r_loss = requery_criterion(v_requery_pred.float(), v_confidence_target)
                                    v_loss = v_a_loss + (v_r_loss * args.requery_weight)
                                    val_loss_total += v_loss.item()
                                    val_batches += 1
                        
                        avg_val_loss = val_loss_total / max(1, val_batches)
                        log.info(f"Validation Loss: {avg_val_loss:.4f}")
                        
                        # Log validation step (re-use previous action/requery loss for continuity in CSV, or None)
                        # We pass None for training components to indicate this is a val update
                        training_logger.log_step(step, epoch, None, None, None, val_loss=avg_val_loss)
                        # Assuming Logger has a generic log_scalar or we just print for now.
                        # logger.writer.add_scalar("Loss/val", avg_val_loss, step) 
                        model.train()
                        
                        # Cleanup validation variables
                        del val_loss_total, val_batches
                        gc.collect()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()

                    # Checkpointing (Step-based)
                    if step % args.save_interval == 0:
                        # Define checkpoint data
                        ckpt_data = {
                            'step': step,
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'ema_state_dict': ema.state_dict(), # Save EMA
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scaler_state_dict': scaler.state_dict(), # Save Scaler
                            'loss': loss.item(),
                        }
                        
                        try:
                            # Try primary save (latest)
                            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                            torch.save(ckpt_data, checkpoint_path)
                        except Exception as e:
                            log.error(f"Failed to save primary checkpoint '{checkpoint_path}': {e}")
                            
                            # Fallback save (timestamped/step-based)
                            try:
                                fallback_name = f"checkpoint_step_{step}.pt"
                                fallback_path = os.path.join(args.checkpoint_dir, fallback_name)
                                log.info(f"Attempting fallback save to '{fallback_path}'...")
                                torch.save(ckpt_data, fallback_path)
                                log.info(f"Fallback save successful!")
                            except Exception as e2:
                                log.error(f"CRITICAL: Fallback save also failed: {e2}")
                        
                        training_logger.plot_progress()
                        
                    # Visualization (Step-based)
                    if args.viz and step % args.viz_interval == 0:
                        visualizer.visualize_batch(step, batch, pred_action)
                
                except torch.cuda.OutOfMemoryError:
                    # CUDA OOM: Reduce batch size
                    consecutive_cuda_oom += 1
                    successful_batches_since_oom = 0
                    
                    log.warning(f"[CUDA OOM] Out of memory at step {step}. (OOM #{consecutive_cuda_oom})")
                    gc.collect()
                    torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    
                    if consecutive_cuda_oom >= max_consecutive_ooms:
                        if effective_batch_size > MIN_BATCH_SIZE:
                            effective_batch_size = max(MIN_BATCH_SIZE, effective_batch_size // 2)
                            consecutive_cuda_oom = 0
                            need_dataloader_rebuild = True
                            log.warning(f"[CUDA OOM] Reducing batch size to {effective_batch_size}")
                            break  # Exit batch loop to rebuild dataloader
                        else:
                            log.error(f"[CUDA OOM] Batch size already at minimum ({MIN_BATCH_SIZE}). Cannot reduce further.")
                            log.error("[CUDA OOM] Saving checkpoint and exiting gracefully...")
                            raise SystemExit("CUDA OOM: Cannot reduce batch size further")
                    
                    continue
                
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # Treat as CUDA OOM
                        consecutive_cuda_oom += 1
                        successful_batches_since_oom = 0
                        
                        log.warning(f"[CUDA OOM] Runtime memory error at step {step}. (OOM #{consecutive_cuda_oom})")
                        gc.collect()
                        torch.cuda.empty_cache()
                        optimizer.zero_grad(set_to_none=True)
                        
                        if consecutive_cuda_oom >= max_consecutive_ooms:
                            if effective_batch_size > MIN_BATCH_SIZE:
                                effective_batch_size = max(MIN_BATCH_SIZE, effective_batch_size // 2)
                                consecutive_cuda_oom = 0
                                need_dataloader_rebuild = True
                                log.warning(f"[CUDA OOM] Reducing batch size to {effective_batch_size}")
                                break
                            else:
                                log.error(f"[CUDA OOM] Batch size already at minimum ({MIN_BATCH_SIZE}). Cannot reduce further.")
                                raise SystemExit("CUDA OOM: Cannot reduce batch size further")
                        
                        continue
                    else:
                        raise
                
                except MemoryError:
                    # System RAM OOM: Reduce shuffle_buffer first, then num_workers
                    consecutive_ram_oom += 1
                    successful_batches_since_oom = 0
                    
                    log.warning(f"[RAM OOM] System memory exhausted at step {step}. (OOM #{consecutive_ram_oom})")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    
                    if consecutive_ram_oom >= max_consecutive_ooms:
                        # First try reducing shuffle buffer
                        if effective_shuffle_buffer > MIN_SHUFFLE_BUFFER:
                            effective_shuffle_buffer = max(MIN_SHUFFLE_BUFFER, effective_shuffle_buffer // 2)
                            consecutive_ram_oom = 0
                            need_dataloader_rebuild = True
                            log.warning(f"[RAM OOM] Reducing shuffle buffer to {effective_shuffle_buffer}")
                            break
                        # Then try reducing workers
                        elif effective_num_workers > MIN_NUM_WORKERS:
                            effective_num_workers = max(MIN_NUM_WORKERS, effective_num_workers - 1)
                            consecutive_ram_oom = 0
                            need_dataloader_rebuild = True
                            log.warning(f"[RAM OOM] Reducing num_workers to {effective_num_workers}")
                            break
                        else:
                            log.error(f"[RAM OOM] All parameters at minimum. Cannot reduce further.")
                            log.error("[RAM OOM] Saving checkpoint and exiting gracefully...")
                            raise SystemExit("RAM OOM: Cannot reduce memory parameters further")
                    
                    continue
                
                # Successful batch - reset OOM counters
                consecutive_cuda_oom = 0
                consecutive_ram_oom = 0
                successful_batches_since_oom += 1
                
                # Auto-recover AR steps after sustained success (disabled for now)
                # if ar_reduction_active and successful_batches_since_oom >= 500:
                #     ...
            
            # Check if we need to rebuild dataloader (from OOM reduction)
            if need_dataloader_rebuild:
                log.info(f"[Memory] Rebuilding dataloader with new parameters...")
                dataloader = create_dataloader(effective_batch_size, effective_shuffle_buffer, effective_num_workers)
                need_dataloader_rebuild = False
                continue  # Restart epoch loop with new dataloader
            
            # Close epoch pbar
            if args.epochs > 0:
                pbar.close()
                steps_per_epoch_actual = steps_in_current_epoch

            # End of Epoch Actions (Epoch-based)
            if args.epochs > 0:
                log.info(f"Saving checkpoint at end of epoch {epoch + 1}")
                torch.save({
                    'step': step,
                    'epoch': epoch + 1, # Save as next epoch start
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                training_logger.plot_progress()
                
                if args.viz:
                    log.info(f"Visualizing at end of epoch {epoch + 1}")
                    # Use the last batch for visualization
                    visualizer.visualize_batch(step, batch, pred_action)

            # End of inner loop
            if args.epochs == 0 and step >= target_step:
                break
            
            # Time limit check break outer
            if args.time_limit_min > 0:
                elapsed_min = (time.time() - start_time) / 60.0
                if elapsed_min >= args.time_limit_min:
                    break

    except KeyboardInterrupt:
        log.info("Training interrupted by user.")
    except Exception as e:
        log.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Archival: Always save a fresh checkpoint to avoid losing data if 'latest' is locked
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}_{args.dataset}_steps{step}.pt"
            run_path = os.path.join(args.checkpoint_dir, run_name)
            
            # Save fresh - do NOT rely on copying 'checkpoint_latest.pt'
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save({
                'step': step,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': loss.item(),
            }, run_path)
            
            log.info(f"Run saved to: {run_path}")
            training_logger.plot_progress()
        except Exception as e:
            log.error(f"Failed to save final checkpoint: {e}")
        
        log.info("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # === Data Source Args ===
    # Option 1: Stream from GCS (default)
    parser.add_argument('--dataset', type=str, default='fractal20220817_data', 
                        help="Dataset name for streaming: droid, fractal20220817_data, etc.")
    parser.add_argument('--data_dir', type=str, default='gs://gresearch/robotics', 
                        help="TFDS data directory for streaming (default: GCS)")
    parser.add_argument('--image_key', type=str, default=None, 
                        help="Image key for streaming (e.g., exterior_image_1_left)")
    
    # Option 2: Load from preprocessed local file (overrides streaming)
    parser.add_argument('--local_data_path', type=str, default=None, 
                        help="Path to preprocessed HDF5 file. If set, uses LocalDROIDLoader instead of streaming.")
    
    # Streaming-specific (ignored when using --local_data_path)
    parser.add_argument('--no_subprocess', action='store_true', 
                        help="[Streaming] Disable subprocess isolation (default: subprocess enabled)")
    parser.add_argument('--shuffle_buffer_size', type=int, default=10, 
                        help="[Streaming] Shuffle buffer size (episodes) for TFDS")
    
    # === Training Args ===
    parser.add_argument('--steps', type=int, default=10000, help="Number of training steps")
    parser.add_argument('--epochs', type=int, default=0, help="Number of epochs (0 = use steps instead)")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--warmup_steps', type=int, default=1000, help="LR warmup steps")
    parser.add_argument('--window_size', type=int, default=10, help="Sliding window size")
    parser.add_argument('--num_workers', type=int, default=0, help="DataLoader workers")
    
    # === Checkpointing ===
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Checkpoint directory")
    parser.add_argument('--save_interval', type=int, default=1000, help="Steps between checkpoints")
    parser.add_argument('--resume', action='store_true', help="Resume from latest checkpoint")
    parser.add_argument('--time_limit_min', type=float, default=0.0, help='Stop after N minutes (0=no limit)')
    
    # === Visualization ===
    parser.add_argument('--viz', action='store_true', help="Enable visualization")
    parser.add_argument('--save_gif', action='store_true', help="Enable GIF generation")
    parser.add_argument('--viz_interval', type=int, default=1000, help="Steps between visualizations")

    # === Loss Function ===
    parser.add_argument('--loss_horizon', type=int, default=1, 
                        help="Twist steps to rollout before loss (1 <= horizon <= window_size)")
    parser.add_argument('--omega_rot', type=float, default=1.0, help="Rotation loss weight")
    parser.add_argument('--omega_trans', type=float, default=1.0, help="Translation loss weight")
    parser.add_argument('--requery_weight', type=float, default=1.0, help="Requery loss weight")
    parser.add_argument('--confidence_temperature', type=float, default=1.0, 
                        help="Temperature for confidence target: exp(-loss/temp)")

    # === Validation ===
    parser.add_argument('--train_split_pct', type=float, default=0.95, help="Train split percentage")
    parser.add_argument('--val_interval', type=int, default=5000, help="Steps between validation")
    parser.add_argument('--val_batch_size', type=int, default=0, help="Validation batch size")
    parser.add_argument('--val_batches', type=int, default=50, help="Batches per validation run")
    
    # === Logging ===
    parser.add_argument('--verbose', type=int, default=2, help="Verbosity: 0=WARNING, 1=INFO, 2=DEBUG")

    args = parser.parse_args()
    
    # === Arg Validation & Conflict Warnings ===
    # Convert no_subprocess to use_subprocess (inverted logic)
    args.use_subprocess = not args.no_subprocess
    
    # Warn about conflicting/ignored args when using local data
    if args.local_data_path:
        ignored_args = []
        
        if args.dataset != 'fractal20220817_data':  # Non-default
            ignored_args.append(f"--dataset={args.dataset}")
        if args.data_dir != 'gs://gresearch/robotics':  # Non-default
            ignored_args.append(f"--data_dir={args.data_dir}")
        if args.image_key is not None:
            ignored_args.append(f"--image_key={args.image_key}")
        if args.no_subprocess:
            ignored_args.append("--no_subprocess")
        if args.shuffle_buffer_size != 10:  # Non-default
            ignored_args.append(f"--shuffle_buffer_size={args.shuffle_buffer_size}")
        
        if ignored_args:
            import warnings
            warnings.warn(
                f"Using --local_data_path, the following streaming args are IGNORED: {', '.join(ignored_args)}"
            )
    
    # Validate loss_horizon
    if args.loss_horizon < 1 or args.loss_horizon > CHUNK_SIZE:
        parser.error(f"--loss_horizon must be between 1 and chunk_size ({CHUNK_SIZE}), got {args.loss_horizon}")
    
    log.debug(f"Args parsed. Viz: {args.viz}, Viz Interval: {args.viz_interval}")
    train(args)
