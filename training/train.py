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
        'num_vision_tokens': 8,
        'latent_dim': 256,
        'queue_size': 10,
        'embed_dim': 256,
        'num_heads': 4,
        'num_layers': 4,
        'robots': {
            'default': {'proprio_dim': 6, 'action_dim': 6}, # Default/UR3e (EE Pose)
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
        
        model.load_state_dict(checkpoint['model_state_dict'])
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
            
        images, proprio, action, goal_embs = batch
        
        # Move to device
        images = images.to(device)
        proprio = proprio.to(device)
        target_action = action.to(device)
        goal_embs = goal_embs.to(device)

        optimizer.zero_grad()
        
        # Note: We use 'default' robot for now. 
        # To support multi-robot, we'd need the dataset to tell us the robot name.
        pred_action, next_latent, requery_logit = model(
            images, 
            proprio, 
            goal_embs, 
            robot_name='default', 
            update_queue=False 
        )
        
        loss = criterion(pred_action, target_action)
        # TODO: Add loss for requery_logit if we have 'is_terminal' or 'done' labels
        # requery_loss = nn.BCEWithLogitsLoss()(requery_logit, target_done)
        # loss += requery_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        step += 1
        pbar.update(1)
        pbar.set_description(f"Loss: {loss.item():.4f}")
        
        if step % args.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fractal20220817_data', help='TFDS Dataset Name')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--embeddings_path', type=str, default='instruction_embeddings.pkl')
    parser.add_argument('--mock', action='store_true', help='Use mock dataset')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory to store/load dataset')
    parser.add_argument('--time_limit_min', type=float, default=0.0, help='Stop training after N minutes')
    
    args = parser.parse_args()
    train(args)
