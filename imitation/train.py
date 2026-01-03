import os
import sys
import gc

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

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.axis_v1 import AxisModel
from imitation.data.rtx_stream_loader import RTXStreamLoader
from imitation.utils.scheduler import CosineAnnealingWarmupRestarts
from imitation.utils.logger import TrainingLogger
from imitation.utils.visualizer import Visualizer
from imitation.utils.ema import EMA
import tensorflow_datasets as tfds
import tensorflow as tf
# Force TensorFlow to use CPU only (prevents VRAM fighting with PyTorch and CUDA errors in workers)
tf.config.set_visible_devices([], 'GPU')

import numpy as np
import torchvision.transforms as T

def generate_episode_gif(model, args, step, visualizer):
    """Loads one episode and generates a GIF."""
    try:
        # Clear cache to free up VRAM for visualization
        torch.cuda.empty_cache()
        
        # Load one episode
        # Use RTXStreamLoader to load one episode safely
        from imitation.data.rtx_stream_loader import RTXStreamLoader
        
        # Manually iterate to get one episode
        if args.data_dir and args.data_dir.startswith('gs://'):
            full_path = f"{args.data_dir}/{args.dataset}/0.1.0"
            builder = tfds.builder_from_directory(builder_dir=full_path)
            ds = builder.as_dataset(split='train', shuffle_files=True)
        else:
            ds = tfds.load(args.dataset, split='train', shuffle_files=True, data_dir=args.data_dir)
            
        # Take one episode
        images_np, proprio_np = None, None
        
        iterator = iter(ds)
        episode = next(iterator)
        
        steps = list(episode['steps'])
        imgs = []
        props = []
        for s in steps:
            img = s['observation']['image']
            img = tf.image.resize(img, (128, 128))
            img = tf.cast(img, tf.float32) / 255.0
            imgs.append(tf.transpose(img, [2, 0, 1]).numpy())
            
            # 8D Pose: [x, y, z, qx, qy, qz, qw, g]
            obs = s['observation']
            p = np.zeros(7, dtype=np.float32)
            if 'base_pose_tool_reached' in obs:
                p = obs['base_pose_tool_reached'].numpy()
            elif 'ee_pose' in obs: p = obs['ee_pose'].numpy()
            elif 'pose' in obs: p = obs['pose'].numpy()
            
            g = 0.0
            if 'gripper_closed' in obs: g = obs['gripper_closed'].numpy()
            elif 'gripper_state' in obs: g = obs['gripper_state'].numpy()
            if not np.isscalar(g): g = g.item() if g.size == 1 else g[0]
            
            p8 = np.zeros(8, dtype=np.float32)
            if p.shape[0] == 7: p8[:7] = p
            elif p.shape[0] == 6: p8[:6] = p
            p8[7] = g
            props.append(p8)
            
        images_np = np.array(imgs)
        proprio_np = np.array(props)
        
        if images_np is None: return

        # Prepare Inputs
        device = next(model.parameters()).device
        T = images_np.shape[0]
        images = torch.tensor(images_np, dtype=torch.float32).to(device).unsqueeze(0)
        proprio = torch.tensor(proprio_np, dtype=torch.float32).to(device).unsqueeze(0)
        
        # Mock Goal (using start/end)
        from imitation.data.goal_oracle import GoalOracle
        oracle = GoalOracle(output_dim=64)
        start_grip = proprio_np[0][7]
        end_grip = proprio_np[-1][7]
        task_type = 0
        if start_grip < 0.5 and end_grip > 0.5: task_type = 1
        elif start_grip > 0.5 and end_grip < 0.5: task_type = 2
        goal_emb = oracle.encode_goal(task_type, proprio_np[0], proprio_np[-1]).to(device).unsqueeze(0)

        # Inference
        pred_actions = []
        requery_preds = []
        model.reset_memory(1)
        W = 8 # window size
        
        with torch.no_grad():
            for t in range(T):
                if t < W:
                    pad_len = W - 1 - t
                    curr_imgs = images[:, :t+1]
                    curr_props = proprio[:, :t+1]
                    pad_imgs = curr_imgs[:, 0:1].repeat(1, pad_len, 1, 1, 1)
                    pad_props = curr_props[:, 0:1].repeat(1, pad_len, 1)
                    win_imgs = torch.cat([pad_imgs, curr_imgs], dim=1)
                    win_props = torch.cat([pad_props, curr_props], dim=1)
                else:
                    win_imgs = images[:, t-W+1 : t+1]
                    win_props = proprio[:, t-W+1 : t+1]
                
                pred_act, _, req_logit = model(win_imgs, win_props, goal_emb, update_queue=True, use_memory=True)
                pred_actions.append(pred_act.cpu())
                requery_preds.append(torch.sigmoid(req_logit).cpu())

        pred_actions = torch.stack(pred_actions, dim=1).squeeze(0)
        requery_preds = torch.stack(requery_preds, dim=1).squeeze(0)
        
        targets = np.zeros_like(proprio_np)
        targets[:-1] = proprio_np[1:]
        targets[-1] = proprio_np[-1]
        targets = torch.tensor(targets, dtype=torch.float32)

        visualizer.create_gif(step, images[0], targets, pred_actions, requery_preds)
        print(f"Generated episode GIF for step {step}", flush=True)
    except Exception as e:
        print(f"Error generating episode GIF: {e}", flush=True)
        import traceback
        traceback.print_exc()

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    print("DEBUG: Script started. Initializing...", flush=True)

    # --- Configuration ---
    config = {
        'device': device,
        'goal_dim': 64, 
        'cond_dim': 256,
        'vision_feature_dim': 256,
        'num_vision_tokens': 8,
        'window_size': 8,
        'latent_dim': 256,
        'queue_size': args.queue_size, 
        'embed_dim': 256,
        'num_heads': 4,
        'num_layers': 4,
        'proprio_dim': 8, # 3 Pos + 4 Quat + 1 Gripper
        'action_dim': 8
    }

    # --- Utilities ---
    logger = TrainingLogger(args.checkpoint_dir)
    visualizer = Visualizer(args.checkpoint_dir)

    # --- Model ---
    model = AxisModel(config).to(device)
    print("Model initialized.")

    # Dataset
    # If epochs > 0, we do NOT repeat the dataset (finite epoch).
    # If steps > 0 (and epochs=0), we repeat the dataset (infinite stream).
    repeat_dataset = (args.epochs == 0)
    
    # === Dynamic Memory Parameters (can be reduced on OOM) ===
    effective_batch_size = args.batch_size
    effective_shuffle_buffer = args.shuffle_buffer_size
    effective_num_workers = args.num_workers
    
    def create_dataloader(batch_size, shuffle_buffer, num_workers, split='train'):
        """Factory function to create/recreate dataloader with specified params."""
        # Split logic: 'train[:95%]' vs 'train[95%:]'
        # Note: shuffle_buffer only applies to training
        is_train = '[:' in split or split == 'train'
        
        print(f"Creating dataloader ({split}): batch_size={batch_size}, shuffle_buffer={shuffle_buffer}, workers={num_workers}")
        stream = RTXStreamLoader(
            dataset_name=args.dataset, 
            split=split, 
            batch_size=1,
            window_size=config['window_size'],
            image_size=(128, 128),
            data_dir=args.data_dir,
            shuffle_buffer_size=shuffle_buffer if is_train else 0,
            repeat=repeat_dataset if is_train else False, # Only repeat train
            max_ar_steps=args.autoregressive_steps
        )
        
        # Handle num_workers=0 case (no prefetch_factor or persistent_workers)
        if num_workers == 0:
            loader = torch.utils.data.DataLoader(
                stream, 
                batch_size=batch_size,
                num_workers=0,
                pin_memory=False
            )
        else:
            loader = torch.utils.data.DataLoader(
                stream, 
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=False, # Disable to save RAM
                prefetch_factor=2, # Reduce to 2 to save RAM (default 4 was too aggressive with large episodes)
                persistent_workers=True
            )
        return loader
    
    # Create initial dataloader
    train_split = f'train[:{int(args.train_split_pct*100)}%]'
    val_split = f'train[{int(args.train_split_pct*100)}%:]'
    
    dataloader = create_dataloader(effective_batch_size, effective_shuffle_buffer, effective_num_workers, split=train_split)
    
    # Validation Dataloader (smaller batch size to save memory if needed, or same)
    # We use 0 workers for val to save overhead unless it's huge
    # LIMIT VAL BATCH SIZE TO 32 to prevent OOM spikes during validation
    val_dataloader = create_dataloader(32, 0, 0, split=val_split)
    
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
    print("EMA initialized.")
    
    # --- Augmentation ---
    # Applied on GPU
    augmentations = nn.Sequential(
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.05),
        # Add slight blur occasionally
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.1)
    ).to(device)
    
    # Requery Loss Function
    # pos_weight allows us to penalize False Negatives (missing help signal) more than False Positives.
    pos_weight = torch.tensor([args.pos_weight]).to(device)
    requery_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Training Loop ---
    model.train()
    
    # --- Resume from Checkpoint ---
    start_step = 0
    start_epoch = 0
    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pt')
    
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming from latest checkpoint: {checkpoint_path}")
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
        print(f"Resumed at step {start_step}, epoch {start_epoch}")
    elif os.path.exists(checkpoint_path):
        print(f"WARNING: Checkpoint found at {checkpoint_path} but --resume was not provided.")
        print("Starting from scratch. Use --resume to continue training.")
    else:
        print("No checkpoint found. Starting from scratch.")

    # Determine total steps/epochs
    if args.epochs > 0:
        total_steps = args.steps # Fallback
        print(f"Training for {args.epochs} epochs.")
    else:
        total_steps = args.steps
        print(f"Training for {args.steps} steps.")

    # Calculate target step (relative to start)
    if args.epochs == 0:
        target_step = start_step + args.steps
    else:
        target_step = float('inf')
    
    print("Starting training...")
    
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
        print("DEBUG: Entering training loop...", flush=True)
        
        # Loop structure
        num_epochs = args.epochs if args.epochs > 0 else 1
        
        # For step-based training, ignore epoch from checkpoint
        # (prevents range(51, 1) = empty when resuming epoch-trained checkpoint)
        if args.epochs == 0:
            start_epoch = 0
        
        # Global pbar for step-based training
        if args.epochs == 0:
            pbar = tqdm(total=target_step, initial=start_step, desc="Training Steps")
        
        steps_per_epoch_actual = None
        
        for epoch in range(start_epoch, num_epochs):
            steps_in_current_epoch = 0  # Initialize for both modes
            
            if args.epochs > 0:
                print(f"--- Epoch {epoch + 1}/{num_epochs} ---")
                
                # Dynamic Progress Bar
                # For the first epoch, we don't know the size, so we use total=None (counter only).
                # For subsequent epochs, we use the number of steps from the previous epoch.
                if epoch == 0:
                    epoch_total = None
                else:
                    epoch_total = steps_per_epoch_actual
                
                pbar = tqdm(total=epoch_total, desc=f"Epoch {epoch + 1}")
            
            for batch in dataloader:
                if args.epochs == 0 and step >= target_step:
                    break
                
                # Check time limit
                if args.time_limit_min > 0:
                    elapsed_min = (time.time() - start_time) / 60.0
                    if elapsed_min >= args.time_limit_min:
                        print(f"\nTime limit of {args.time_limit_min} minutes reached. Stopping.")
                        break
                
                # === OOM Protection: Wrap batch processing in try/except ===
                try:
                    # Periodic memory cleanup (every 100 steps)
                    if step % 100 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # Unpack Batch
                    images = batch['images'].to(device)   # (B, 8, 3, 128, 128)
                    proprio = batch['proprio'].to(device) # (B, 8, 8)
                    goal_embs = batch['goal'].to(device)  # (B, 64)
                    target_actions = batch['actions'].to(device) # (B, N, 8) - N sequential targets
                    target_requery = batch['requery'].to(device) # (B, 1)

                    B = images.shape[0]
                    W = images.shape[1]  # Window size

                    # === Data Augmentation ===
                    if model.training:
                        # Reshape to (B*T, C, H, W) for efficient augmentation
                        fl_imgs = images.view(-1, 3, 128, 128)
                        aug_imgs = augmentations(fl_imgs)
                        images = aug_imgs.view(B, W, 3, 128, 128)

                    optimizer.zero_grad()
                    model.reset_memory(B) 
                    
                    # === Forward Pass with AMP ===
                    with torch.cuda.amp.autocast():
                        current_images = images.clone()
                        current_proprio = proprio.clone()
                        
                        total_action_loss = 0.0
                        total_requery_loss = 0.0
                        
                        for ar_step in range(args.autoregressive_steps):
                            # Forward Pass
                            pred_action, next_latent, requery_logit = model(
                                current_images, 
                                current_proprio, 
                                goal_embs, 
                                update_queue=args.use_latent_memory,
                                use_memory=args.use_latent_memory
                            )
                            
                            # Compute Loss for this step
                            # Each AR step uses its temporally-aligned target
                            target_action = target_actions[:, ar_step, :]  # (B, 8)
                            step_action_loss = torch.mean((pred_action - target_action)**2)
                            
                            if ar_step == 0:
                                step_requery_loss = requery_criterion(requery_logit, target_requery)
                                total_action_loss = step_action_loss
                                total_requery_loss = step_requery_loss
                            else:
                                # Later steps contribute with decay
                                total_action_loss = total_action_loss + step_action_loss * (args.ar_loss_decay ** ar_step)
                            
                            # Prepare next iteration: shift window and inject prediction
                            if ar_step < args.autoregressive_steps - 1:
                                current_images = torch.cat([
                                    current_images[:, 1:, :, :, :],
                                    images[:, -1:, :, :, :]  # Use original GT for last frame
                                ], dim=1)
                                
                                current_proprio = torch.cat([
                                    current_proprio[:, 1:, :],
                                    pred_action.unsqueeze(1)  # (B, 1, 8)
                                ], dim=1)
                        
                        # Final losses (average over steps if > 1)
                        action_loss = total_action_loss / args.autoregressive_steps
                        requery_loss = total_requery_loss  # Only computed once
                        
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
                    logger.log_step(step, epoch, loss.item(), action_loss.item(), requery_loss.item())
                    
                    step += 1
                    steps_in_current_epoch += 1
                    pbar.update(1)
                    pbar.set_description(f"L:{loss.item():.4f} A:{action_loss.item():.4f} R:{requery_loss.item():.4f}")
                    
                    # === Validation Loop ===
                    if step % args.val_interval == 0:
                        # Force GC to clear any transient training memory
                        gc.collect()
                        if torch.cuda.is_available(): os.environ.get('empty_cache', torch.cuda.empty_cache)()

                        model.eval()
                        val_loss_total = 0.0
                        val_batches = 0
                        print("\nRunning Validation...", flush=True)
                        with torch.no_grad():
                            for val_batch in val_dataloader:
                                if val_batches >= args.val_batches: break # Limit val batches
                                
                                v_imgs = val_batch['images'].to(device)
                                v_props = val_batch['proprio'].to(device)
                                v_goals = val_batch['goal'].to(device)
                                v_actions = val_batch['actions'].to(device)
                                v_requery = val_batch['requery'].to(device)
                                
                                B_val = v_imgs.shape[0]
                                model.reset_memory(B_val)
                                
                                with torch.cuda.amp.autocast():
                                    # Non-AR validation for speed/standard metric (or should it be AR?)
                                    # Let's do AR=1 for standard validation
                                    v_pred, _, v_req_logit = model(v_imgs, v_props, v_goals, update_queue=False, use_memory=True)
                                    
                                    v_target_act = v_actions[:, 0, :]
                                    v_a_loss = torch.mean((v_pred - v_target_act)**2)
                                    v_r_loss = requery_criterion(v_req_logit, v_requery)
                                    v_loss = v_a_loss + (v_r_loss * args.requery_weight)
                                    val_loss_total += v_loss.item()
                                    val_batches += 1
                        
                        avg_val_loss = val_loss_total / max(1, val_batches)
                        print(f"Validation Loss: {avg_val_loss:.4f}", flush=True)
                        # Log validation loss (You might need to update Logger to handle 'val_loss' key or just print)
                        # Assuming Logger has a generic log_scalar or we just print for now.
                        # logger.writer.add_scalar("Loss/val", avg_val_loss, step) 
                        model.train()
                        
                        # Cleanup validation variables
                        del val_loss_total, val_batches
                        gc.collect()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()

                    # Checkpointing (Step-based)
                    if args.epochs == 0 and step % args.save_interval == 0:
                        torch.save({
                            'step': step,
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'ema_state_dict': ema.state_dict(), # Save EMA
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scaler_state_dict': scaler.state_dict(), # Save Scaler
                            'loss': loss.item(),
                        }, checkpoint_path)
                        logger.plot_progress()
                        
                    # Visualization (Step-based)
                    if args.epochs == 0 and args.viz and step % args.viz_interval == 0:
                        visualizer.visualize_batch(step, batch, pred_action)
                        if args.save_gif:
                            generate_episode_gif(model, args, step, visualizer)
                
                except torch.cuda.OutOfMemoryError:
                    # CUDA OOM: Reduce batch size
                    consecutive_cuda_oom += 1
                    successful_batches_since_oom = 0
                    
                    print(f"\n[CUDA OOM] Out of memory at step {step}. (OOM #{consecutive_cuda_oom})")
                    gc.collect()
                    torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    
                    if consecutive_cuda_oom >= max_consecutive_ooms:
                        if effective_batch_size > MIN_BATCH_SIZE:
                            effective_batch_size = max(MIN_BATCH_SIZE, effective_batch_size // 2)
                            consecutive_cuda_oom = 0
                            need_dataloader_rebuild = True
                            print(f"[CUDA OOM] Reducing batch size to {effective_batch_size}")
                            break  # Exit batch loop to rebuild dataloader
                        else:
                            print(f"[CUDA OOM] Batch size already at minimum ({MIN_BATCH_SIZE}). Cannot reduce further.")
                            print("[CUDA OOM] Saving checkpoint and exiting gracefully...")
                            raise SystemExit("CUDA OOM: Cannot reduce batch size further")
                    
                    continue
                
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # Treat as CUDA OOM
                        consecutive_cuda_oom += 1
                        successful_batches_since_oom = 0
                        
                        print(f"\n[CUDA OOM] Runtime memory error at step {step}. (OOM #{consecutive_cuda_oom})")
                        gc.collect()
                        torch.cuda.empty_cache()
                        optimizer.zero_grad(set_to_none=True)
                        
                        if consecutive_cuda_oom >= max_consecutive_ooms:
                            if effective_batch_size > MIN_BATCH_SIZE:
                                effective_batch_size = max(MIN_BATCH_SIZE, effective_batch_size // 2)
                                consecutive_cuda_oom = 0
                                need_dataloader_rebuild = True
                                print(f"[CUDA OOM] Reducing batch size to {effective_batch_size}")
                                break
                            else:
                                print(f"[CUDA OOM] Batch size already at minimum ({MIN_BATCH_SIZE}). Cannot reduce further.")
                                raise SystemExit("CUDA OOM: Cannot reduce batch size further")
                        
                        continue
                    else:
                        raise
                
                except MemoryError:
                    # System RAM OOM: Reduce shuffle_buffer first, then num_workers
                    consecutive_ram_oom += 1
                    successful_batches_since_oom = 0
                    
                    print(f"\n[RAM OOM] System memory exhausted at step {step}. (OOM #{consecutive_ram_oom})")
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
                            print(f"[RAM OOM] Reducing shuffle buffer to {effective_shuffle_buffer}")
                            break
                        # Then try reducing workers
                        elif effective_num_workers > MIN_NUM_WORKERS:
                            effective_num_workers = max(MIN_NUM_WORKERS, effective_num_workers - 1)
                            consecutive_ram_oom = 0
                            need_dataloader_rebuild = True
                            print(f"[RAM OOM] Reducing num_workers to {effective_num_workers}")
                            break
                        else:
                            print(f"[RAM OOM] All parameters at minimum. Cannot reduce further.")
                            print("[RAM OOM] Saving checkpoint and exiting gracefully...")
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
                print(f"[Memory] Rebuilding dataloader with new parameters...")
                dataloader = create_dataloader(effective_batch_size, effective_shuffle_buffer, effective_num_workers)
                need_dataloader_rebuild = False
                continue  # Restart epoch loop with new dataloader
            
            # Close epoch pbar
            if args.epochs > 0:
                pbar.close()
                steps_per_epoch_actual = steps_in_current_epoch

            # End of Epoch Actions (Epoch-based)
            if args.epochs > 0:
                print(f"Saving checkpoint at end of epoch {epoch + 1}")
                torch.save({
                    'step': step,
                    'epoch': epoch + 1, # Save as next epoch start
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                logger.plot_progress()
                
                if args.viz:
                    print(f"Visualizing at end of epoch {epoch + 1}")
                    # Use the last batch for visualization
                    visualizer.visualize_batch(step, batch, pred_action)
                    if args.save_gif:
                        generate_episode_gif(model, args, step, visualizer)

            # End of inner loop
            if args.epochs == 0 and step >= target_step:
                break
            
            # Time limit check break outer
            if args.time_limit_min > 0:
                elapsed_min = (time.time() - start_time) / 60.0
                if elapsed_min >= args.time_limit_min:
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Archival
        if os.path.exists(checkpoint_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}_{args.dataset}_steps{step}.pt"
            run_path = os.path.join(args.checkpoint_dir, run_name)
            shutil.copy(checkpoint_path, run_path)
            print(f"\nRun saved to: {run_path}")
            logger.plot_progress()
            print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Training Args
    parser.add_argument('--dataset', type=str, default='fractal20220817_data', help="Dataset name")
    parser.add_argument('--data_dir', type=str, default='gs://gresearch/robotics', help="Data directory (local or GCS)")
    parser.add_argument('--steps', type=int, default=10000, help="Number of training steps")
    parser.add_argument('--epochs', type=int, default=0, help="Number of training epochs (overrides steps if > 0)")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--warmup_steps', type=int, default=1000, help="Number of warmup steps for LR scheduler")
    parser.add_argument('--save_interval', type=int, default=1000, help="Interval to save checkpoints")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Checkpoint directory")
    parser.add_argument('--resume', action='store_true', help="Resume from latest checkpoint")
    parser.add_argument('--time_limit_min', type=float, default=0.0, help='Stop training after N minutes')
    
    # New Arguments
    parser.add_argument('--use_latent_memory', action='store_true', help="Enable recurrent latent memory updates")
    parser.add_argument('--queue_size', type=int, default=10, help="Size of the latent memory queue")
    
    # Visualization Args
    parser.add_argument('--viz', action='store_true', help="Enable visualization during training")
    parser.add_argument('--save_gif', action='store_true', help="Enable GIF generation (memory intensive)")
    parser.add_argument('--viz_interval', type=int, default=1000, help="Step interval for visualization")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of dataloader workers")
    parser.add_argument('--shuffle_buffer_size', type=int, default=10, help="Shuffle buffer size (episodes) for TFDS (keep low for memory!)")
    parser.add_argument('--requery_weight', type=float, default=1.0, help="Weight for requery loss (increase to 10.0+ to prioritize asking for help)")
    parser.add_argument('--pos_weight', type=float, default=1.0, help="Asymmetric loss weight for positive class (increase to >1.0 to penalize missing 'help needed' signals)")
    
    # Autoregressive Training Args
    parser.add_argument('--autoregressive_steps', type=int, default=1, help="Number of autoregressive rollout steps during training. 1=standard, >1=predict and feed back before backprop")
    parser.add_argument('--ar_loss_decay', type=float, default=0.5, help="Decay factor for loss on subsequent autoregressive steps (0.5 = each step contributes half as much)")

    # Validation Args
    parser.add_argument('--train_split_pct', type=float, default=0.95, help="Percentage of data to use for training (remainder for validation)")
    parser.add_argument('--val_interval', type=int, default=5000, help="Step interval for validation")
    parser.add_argument('--val_batches', type=int, default=50, help="Number of batches to run during validation")

    args = parser.parse_args()
    print(f"DEBUG: Args parsed. Viz: {args.viz}, Viz Interval: {args.viz_interval}", flush=True)
    train(args)
