
import os
import sys
import os
import sys
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

from training.model.axis_v1 import AxisModel
from training.data.rtx_stream_loader import RTXStreamLoader
from training.utils.scheduler import CosineAnnealingWarmupRestarts
from training.utils.logger import TrainingLogger
from training.utils.visualizer import Visualizer

def train(args):
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
        'queue_size': args.queue_size, 
        'embed_dim': 256,
        'num_heads': 4,
        'num_layers': 4,
        'proprio_dim': 7,
        'action_dim': 7
    }

    # --- Utilities ---
    logger = TrainingLogger(args.checkpoint_dir)
    visualizer = Visualizer(args.checkpoint_dir)

    # --- Model ---
    model = AxisModel(config).to(device)
    print("Model initialized.")

    # Dataset
    # RTXStreamLoader yields individual windows.
    # We use DataLoader to batch them.
    print(f"Initializing RTXStreamLoader for {args.dataset}...")
    stream_loader = RTXStreamLoader(
        dataset_name=args.dataset, 
        split='train', 
        batch_size=1, # Loader yields 1 window at a time
        window_size=config['window_size'],
        image_size=(128, 128),
        data_dir=args.data_dir
    )
    
    # Wrap in DataLoader for batching
    # Note: num_workers=0 is usually safer for streaming datasets to avoid forking issues with GCS
    dataloader = torch.utils.data.DataLoader(
        stream_loader, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    # --- Optimizer & Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer, 
        first_cycle_steps=args.steps, 
        max_lr=args.lr, 
        min_lr=1e-6, 
        warmup_steps=args.warmup_steps
    )
    
    # Requery Loss Function
    requery_criterion = nn.BCEWithLogitsLoss()

    # --- Training Loop ---
    model.train()
    
    # --- Resume from Checkpoint ---
    start_step = 0
    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pt')
    
    if os.path.exists(checkpoint_path):
        print(f"Resuming from latest checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Filter state dict
        state_dict = checkpoint['model_state_dict']
        model_state = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
                
        model.load_state_dict(filtered_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        print(f"Resumed at step {start_step}")
    else:
        print("No checkpoint found. Starting from scratch.")

    print("Starting training...")
    pbar = tqdm(total=args.steps, initial=start_step)
    
    import time
    start_time = time.time()
    step = start_step 
    
    try:
        # Iterate over batches
        for batch in dataloader:
            if step >= args.steps:
                break
            
            # Check time limit
            if args.time_limit_min > 0:
                elapsed_min = (time.time() - start_time) / 60.0
                if elapsed_min >= args.time_limit_min:
                    print(f"\nTime limit of {args.time_limit_min} minutes reached. Stopping.")
                    break
                
            # Unpack Batch
            # RTXStreamLoader yields: {'images', 'proprio', 'goal', 'action', 'requery'}
            images = batch['images'].to(device)   # (B, 8, 3, 128, 128)
            proprio = batch['proprio'].to(device) # (B, 8, 7)
            goal_embs = batch['goal'].to(device)  # (B, 64)
            target_action = batch['action'].to(device) # (B, 7)
            target_requery = batch['requery'].to(device) # (B, 1)

            B = images.shape[0]

            optimizer.zero_grad()
            
            # Reset Latent Queue for the new batch?
            # Note: In this streaming window approach, each batch is independent. 
            # We don't maintain state across batches for the same episode because batches are shuffled.
            # So we should reset memory for every batch.
            model.reset_memory(B) 
            
            # Forward Pass
            # Model takes (B, W, ...) and outputs prediction for the LAST step in the window.
            pred_action, next_latent, requery_logit = model(
                images, 
                proprio, 
                goal_embs, 
                update_queue=args.use_latent_memory,
                use_memory=args.use_latent_memory
            )
            
            # Compute Action Loss (MSE)
            action_loss = torch.mean((pred_action - target_action)**2)
            
            # Compute Requery Loss (BCE)
            requery_loss = requery_criterion(requery_logit, target_requery)
            
            # Combine
            loss = action_loss + requery_loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Logging
            logger.log_step(step, loss.item(), action_loss.item(), requery_loss.item())
            
            step += 1
            pbar.update(1)
            pbar.set_description(f"L:{loss.item():.4f} A:{action_loss.item():.4f} R:{requery_loss.item():.4f}")
            
            # Checkpointing & Visualization
            if step % args.save_interval == 0:
                # Save Latest Checkpoint
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                
                # Update Plot
                logger.plot_progress()
                
                # Visualization (if enabled and interval met)
                if args.viz and step % args.viz_interval == 0:
                    visualizer.visualize_batch(step, batch, pred_action)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Archival: Rename latest checkpoint to run checkpoint
        if os.path.exists(checkpoint_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}_{args.dataset}_steps{step}.pt"
            run_path = os.path.join(args.checkpoint_dir, run_name)
            shutil.copy(checkpoint_path, run_path)
            print(f"\nRun saved to: {run_path}")
            
            # Also save final plot
            logger.plot_progress()
            print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='TFDS Dataset Name')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory to store/load dataset')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--time_limit_min', type=float, default=0.0, help='Stop training after N minutes')
    
    # New Arguments
    parser.add_argument('--use_latent_memory', action='store_true', help="Enable recurrent latent memory updates")
    parser.add_argument('--queue_size', type=int, default=10, help="Size of the latent memory queue")
    
    # Visualization Args
    parser.add_argument('--viz', action='store_true', help="Enable visualization during training")
    parser.add_argument('--viz_interval', type=int, default=1000, help="Step interval for visualization")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of dataloader workers")

    args = parser.parse_args()
    train(args)
