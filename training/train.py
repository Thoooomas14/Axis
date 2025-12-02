
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import pickle
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.model.axis_v1 import AxisModel
from training.data.rtx_loader import RTXDataset
from training.data.mock_loader import MockDataset
from training.utils.scheduler import CosineAnnealingWarmupRestarts

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Configuration ---
    # In a real project, load this from a yaml file
    config = {
        'device': device,
        'goal_dim': 77, # Structured Gemini Goal
        'cond_dim': 256,
        'vision_feature_dim': 256,
        'num_vision_tokens': 1, # User requested 1 token per image
        'window_size': 8, # User requested 8 time steps context
        'latent_dim': 256,
        'queue_size': args.queue_size, # Use argument for queue_size
        'embed_dim': 256,
        'num_heads': 4,
        'num_layers': 4,
        'robots': {
            'default': {'proprio_dim': 7, 'action_dim': 7}, # Default/UR3e (EE Pose + Gripper)
            # Add other robots here if training on mixed data
            # 'widowx': {'proprio_dim': 6, 'action_dim': 6}
        }
    }


    # --- Model ---
    model = AxisModel(config).to(device)
    print("Model initialized.")

    # --- Data ---
    # Load pre-computed embeddings
    if os.path.exists(args.embeddings_path):
        with open(args.embeddings_path, 'rb') as f:
            instruction_embeddings = pickle.load(f)
        print(f"Loaded {len(instruction_embeddings)} instruction embeddings.")
    else:
        # print(f"Warning: Embeddings file not found at {args.embeddings_path}. Using random embeddings.")
        instruction_embeddings = {}

    # Dataset
    if args.mock:
        print("Using MOCK Dataset.")
        dataset = MockDataset(
            batch_size=args.batch_size,
            image_size=(128, 128),
            length=args.steps * args.batch_size # Ensure enough data
        )
    elif args.use_processed_data:
        print(f"Using Processed Dataset from {args.data_dir}")
        from training.data.processed_loader import ProcessedDataset
        # ProcessedDataset is a map-style dataset, so we wrap it in a DataLoader
        # But train.py expects an iterable.
        # We can use DataLoader(..., shuffle=True) and iterate it.
        
        ds = ProcessedDataset(data_dir=args.data_dir)
        # Create DataLoader
        from torch.utils.data import DataLoader
        dataset = DataLoader(
            ds, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers, 
            pin_memory=True,
            drop_last=True
        )
    else:
        # Note: We assume the dataset yields (image, proprio, instruction_text)
        dataset = RTXDataset(
            dataset_name=args.dataset, 
            split='train', 
            batch_size=args.batch_size,
            image_size=(128, 128),
            data_dir=args.data_dir
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
    
    criterion = nn.MSELoss()

    # --- Training Loop ---
    model.train()
    step = 0
    
    # Create checkpoint dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- Resume from Checkpoint ---
    start_step = 0
    # Find latest checkpoint
    checkpoints = [f for f in os.listdir(args.checkpoint_dir) if f.endswith('.pt')]
    if checkpoints:
        # Sort by step number (checkpoint_1000.pt)
        checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        latest_checkpoint = checkpoints[-1]
        checkpoint_path = os.path.join(args.checkpoint_dir, latest_checkpoint)
        
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Filter out latent_queue state if shapes don't match
        state_dict = checkpoint['model_state_dict']
        model_state = model.state_dict()
        
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state:
                if v.shape != model_state[k].shape:
                    print(f"Skipping {k} due to shape mismatch: {v.shape} vs {model_state[k].shape}")
                    continue
                filtered_state_dict[k] = v
                
        model.load_state_dict(filtered_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        print(f"Resumed at step {start_step}")
    else:
        print("No checkpoint found. Starting from scratch.")

    print("Starting training...")
    # We iterate by steps since RLDS datasets can be infinite or huge
    pbar = tqdm(total=args.steps, initial=start_step)
    
    import time
    start_time = time.time()

    step = start_step # Initialize step counter
    
    for batch in dataset:
        if step >= args.steps:
            break
        
        # Check time limit
        if args.time_limit_min > 0:
            elapsed_min = (time.time() - start_time) / 60.0
            if elapsed_min >= args.time_limit_min:
                print(f"\nTime limit of {args.time_limit_min} minutes reached. Stopping.")
                break
            
        images, proprio, action, goal_embs, mask, requery_labels = batch
        
        # Move to device
        # Shape: (B, T, ...)
        images = images.to(device)
        proprio = proprio.to(device)
        target_action = action.to(device)
        goal_embs = goal_embs.to(device) # (B, T, 77)
        mask = mask.to(device) # (B, T)
        requery_labels = requery_labels.to(device) # (B, T, 1)

        B, T = images.shape[0], images.shape[1]

        optimizer.zero_grad()
        
        # Reset Latent Queue for the new batch (Start of Episode)
        model.reset_memory(B) 
        
        total_loss = 0
        total_action_loss = 0
        total_requery_loss = 0
        valid_steps = 0
        
        # Requery Loss Function
        requery_criterion = nn.BCEWithLogitsLoss(reduction='none')

        # Loop over time
        # We need a window of context (e.g. 8 steps)
        W_size = config.get('window_size', 8)
        
        for t in range(T):
            # Define window range [t_start, t_end] (inclusive of t)
            if t < W_size - 1:
                # Need padding
                pad_len = W_size - 1 - t
                
                img_slice = images[:, :t+1]
                prop_slice = proprio[:, :t+1]
                
                img_pad = img_slice[:, 0:1].repeat(1, pad_len, 1, 1, 1)
                prop_pad = prop_slice[:, 0:1].repeat(1, pad_len, 1)
                
                img_window = torch.cat([img_pad, img_slice], dim=1)
                prop_window = torch.cat([prop_pad, prop_slice], dim=1)
            else:
                # Full window available
                img_window = images[:, t-W_size+1 : t+1]
                prop_window = proprio[:, t-W_size+1 : t+1]
            
            # Targets for current step t
            act_t = target_action[:, t] # (B, 6)
            mask_t = mask[:, t] # (B,)
            goal_t = goal_embs[:, t] # (B, 77) - Current Goal
            requery_t = requery_labels[:, t] # (B, 1)
            
            # Forward Pass
            pred_action, next_latent, requery_logit = model(
                img_window, 
                prop_window, 
                goal_t, 
                robot_name='default', 
                update_queue=args.use_latent_memory,
                use_memory=args.use_latent_memory
            )
            
            # Compute Action Loss (MSE)
            # Masked mean
            action_mse = torch.mean((pred_action - act_t)**2, dim=-1) # (B,)
            masked_action_loss = (action_mse * mask_t).sum() / (mask_t.sum() + 1e-6)
            
            # Compute Requery Loss (BCE)
            requery_bce = requery_criterion(requery_logit, requery_t) # (B, 1)
            requery_bce = requery_bce.squeeze(-1) # (B,)
            masked_requery_loss = (requery_bce * mask_t).sum() / (mask_t.sum() + 1e-6)
            
            # Combine
            step_loss = masked_action_loss + masked_requery_loss
            
            total_loss += step_loss
            total_action_loss += masked_action_loss
            total_requery_loss += masked_requery_loss
            valid_steps += 1
            
        # Average loss over time
        loss = total_loss / T
        avg_act_loss = total_action_loss / T
        avg_req_loss = total_requery_loss / T
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        step += 1
        pbar.update(1)
        pbar.set_description(f"L:{loss.item():.4f} A:{avg_act_loss.item():.4f} R:{avg_req_loss.item():.4f}")
        
        if step % args.save_interval == 0:
            save_path = os.path.join(args.checkpoint_dir, f"checkpoint_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, save_path)
            
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
    parser.add_argument('--embeddings_path', type=str, default='instruction_embeddings.pkl')
    parser.add_argument('--mock', action='store_true', help='Use mock dataset')
    parser.add_argument('--time_limit_min', type=float, default=0.0, help='Stop training after N minutes')
    
    # New Arguments
    parser.add_argument('--use_latent_memory', action='store_true', help="Enable recurrent latent memory updates during training")
    parser.add_argument('--queue_size', type=int, default=10, help="Size of the latent memory queue")
    parser.add_argument('--use_processed_data', action='store_true', help="Use preprocessed .pt files")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of dataloader workers")

    args = parser.parse_args()
    train(args)
