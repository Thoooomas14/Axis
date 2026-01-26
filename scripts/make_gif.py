import os
import sys
import torch
import argparse
import numpy as np
import argparse
import numpy as np

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow_datasets as tfds
import tensorflow as tf
from scipy.spatial.transform import Rotation as R

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.axis import AxisModel
from imitation.utils.visualizer import Visualizer
from imitation.data.goal_oracle import GoalOracle
from imitation.data.rtx_stream_loader import RTXStreamLoader
from imitation.data.local_loader import LocalDataLoader

def make_gif(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- V2 Configuration ---
    config = {
        'device': device,
        'goal_dim': 38, 
        'embed_dim': 256,
        'num_heads': 4,
        'num_layers': 4,
        'proprio_dim': 13, # 13D Pose
        'action_dim': 7,   # 7D Twist
        'use_rope': True,
        'chunk_size': 10
    }

    # --- Load Model ---
    model = AxisModel(config).to(device)
    
    if args.random_weights:
        print("WARNING: Using Random Weights.")
        model.eval()
    else:
        model.eval()
        checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pt')
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            # Try finding runs
            if os.path.exists(args.checkpoint_dir):
                files = [f for f in os.listdir(args.checkpoint_dir) if f.endswith('.pt')]
                if files:
                    files.sort(key=lambda x: os.path.getmtime(os.path.join(args.checkpoint_dir, x)))
                    checkpoint_path = os.path.join(args.checkpoint_dir, files[-1])
                    print(f"Using latest found checkpoint: {checkpoint_path}")
                else:
                    print("No checkpoints found.")
                    return
            else:
                 print(f"Checkpoint dir {args.checkpoint_dir} does not exist.")
                 return

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model at step {checkpoint.get('step', 'unknown')}")

    # --- Load Data ---
    if args.local_data_path:
        print(f"Loading from LOCAL HDF5: {args.local_data_path}")
        loader = LocalDataLoader(
            data_path=args.local_data_path,
            window_size=8,
            loss_horizon=10,
            shuffle=False,
            repeat=False,
            max_episodes=args.max_episodes if args.max_episodes > 0 else 1,
        )
    else:
        print(f"Initializing RTXStreamLoader for {args.dataset}...")
        loader = RTXStreamLoader(
            dataset_name=args.dataset,
            split='train',
            batch_size=1,
            window_size=8,
            image_size=(128, 128),
            data_dir=args.data_dir,
            repeat=False,
            use_subprocess=False,
            shuffle_buffer_size=0,
            shuffle_files=False,
            max_episodes=args.max_episodes if args.max_episodes > 0 else 1,
        )
    
    iterator = iter(loader)

    # --- Visualization Setup ---
    viz = Visualizer(args.checkpoint_dir)

    # --- Inference Loop ---
    pred_actions = []
    requery_preds = []
    images_collected = []
    gt_twists = []
    
    # Store Poses
    gt_poses = []
    pred_poses = []
    
    inference_times = []
    
    print(f"Processing {args.length} steps...")
    
    # Initialize predicted pose with first GT pose
    # We need to get the first batch to initialize this.
    # But iterator gives batches one by one.
    # We'll initialize on the first step.
    curr_pred_pose_13d = None
    current_proprio_window = None
    
    with torch.no_grad():
        for i in range(args.length):
            try:
                batch = next(iterator)
            except StopIteration:
                print("Dataset exhausted.")
                break
                
            if i % 10 == 0: print(f"Step {i}/{args.length}")

            # Unpack Batch
            images = batch['images'].to(device)
            if images.dim() == 4: images = images.unsqueeze(0)
                
            proprio = batch['proprio'].to(device)
            if proprio.dim() == 2: proprio = proprio.unsqueeze(0)
                
            goal = batch['goal'].to(device)
            if goal.dim() == 1: goal = goal.unsqueeze(0)
            
            # Initialize Prediction state if needed
            if curr_pred_pose_13d is None:
                # Start from the LAST proprio of the FIRST window?
                # Actually, make_gif usually iterates sequential windows.
                # The "current" state is the last item in the window input.
                curr_pred_pose_13d = proprio[0, -1].cpu().numpy() # (13,)
                
                # Also store initial poses
                pred_poses.append(curr_pred_pose_13d)
            
            # Collect GT Pose (current state at end of window)
            gt_pose_curr = proprio[0, -1].cpu().numpy() # (13,)
            gt_poses.append(gt_pose_curr)

            # Determine Proprio Input
            if args.closed_loop:
                if current_proprio_window is None:
                    # Initialize with GT from first batch
                    current_proprio_window = proprio.clone()
                proprio_input = current_proprio_window
            else:
                proprio_input = proprio
            
            # Forward Pass
            import time
            start_time = time.time()
            output = model(images, proprio_input, goal)
            
            if isinstance(output, tuple):
                 pred_chunk, req_logit = output[0], output[1]
            else:
                 pred_chunk = output
                 req_logit = torch.zeros(1, 1).to(device)
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            # Action: (1, Chunk, 7). Take first step.
            current_pred_action = pred_chunk[:, 0, :].cpu() # (1, 7)
            current_req = torch.sigmoid(req_logit).cpu() # (1, 1)
            
            pred_actions.append(current_pred_action)
            requery_preds.append(current_req)
            
            # Integrate Prediction
            # Update curr_pred_pose_13d from PREVIOUS PREDICTED STATE using predicted action
            # This ensures we visualize the smooth accumulated trajectory (rollout),
            # even if the model input (proprio_input) was reset to GT (Teacher Forcing).
            integration_start_state = curr_pred_pose_13d
            
            # integrate_twist returns (H, 13), we pass (1, 7)
            integrated = viz.integrate_twist(integration_start_state, current_pred_action)
            next_pred_pose = integrated[0]
            pred_poses.append(next_pred_pose)
            
            # Update state for next step
            curr_pred_pose_13d = next_pred_pose
            
            if args.closed_loop:
                # Update window for next step
                # Shift left and append new pose
                next_pose_t = torch.tensor(next_pred_pose, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                current_proprio_window = torch.cat([current_proprio_window[:, 1:, :], next_pose_t], dim=1)
            
            # Collect Visualization Data (Image)
            last_img = images[0, -1].cpu() # (3, H, W)
            images_collected.append(last_img)
            
            # GT Twist (Future)
            if 'actions' in batch:
                gt = batch['actions'][0, 0].cpu() # (7,)
                gt_twists.append(gt)
            else:
                gt_twists.append(torch.zeros(7))
    
    if not images_collected:
        print("No data collected.")
        return

    # Stack results
    pred_actions = torch.cat(pred_actions, dim=0) 
    requery_preds = torch.cat(requery_preds, dim=0)
    images_stack = torch.stack(images_collected, dim=0) 
    gt_stack = torch.stack(gt_twists, dim=0)
    
    # Poses
    # gt_poses has T entries. pred_poses has T+1 entries (initial + T updates).
    # We align them: pred_poses[0] is initial (same as gt_poses[0] roughly).
    # Actually, iterate loop does T steps.
    # gt_poses collected T times (current state).
    # pred_poses collected T+1 times (initial + T nexts).
    # visualizing step t:
    # Image[t], Twist[t], Pose[t] (state AT t)
    # We should match lengths.
    gt_poses_stack = torch.tensor(np.array(gt_poses), dtype=torch.float32)
    # Take first T predicted poses to match image count?
    # Or start from 1?
    # pred_poses[0] is state at start of step 0.
    # pred_poses[1] is state at start of step 1 (result of action 0).
    # We want to plot the trajectory. Matching lengths is best.
    pred_poses_stack = torch.tensor(np.array(pred_poses[:-1]), dtype=torch.float32)
    
    print(f"Collected {len(images_collected)} frames. Poses: {gt_poses_stack.shape}")

    # --- Generate GIF ---
    print("Generating GIF...")
    viz.create_gif(
        0, 
        images_stack, 
        gt_stack, 
        pred_actions,
        gt_poses=gt_poses_stack,
        pred_poses=pred_poses_stack,
        requery_preds=requery_preds, 
        inference_times=inference_times, 
        save_prefix='episode_stream_viz', 
        dpi=100
    )
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='droid')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--local_data_path', type=str, default=None,
                        help='Path to local HDF5 file (overrides streaming)')
    parser.add_argument('--max_episodes', type=int, default=1,
                        help='Max episodes to use (default: 1 for overfit testing)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--random_weights', action='store_true', help="Use random weights")
    parser.add_argument('--closed_loop', action='store_true', help="Use closed-loop proprioception feedback")
    parser.add_argument('--length', type=int, default=100, help="Number of steps to visualize")
    
    args = parser.parse_args()
    make_gif(args)
