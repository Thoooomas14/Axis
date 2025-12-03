import os
import sys
import torch
import argparse
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.model.axis_v1 import AxisModel
from training.utils.visualizer import Visualizer
# We need a way to load full episodes. 
# Since RTXStreamLoader yields windows, we can't easily use it for full episodes.
# We will use a simplified dataset class here that loads one episode.
# Or we can use the old RTXDataset logic if it still works, or just implement a simple loader here.

import tensorflow_datasets as tfds
import tensorflow as tf

def load_episode(dataset_name, data_dir, split='train'):
    """Loads a single random episode from the dataset."""
    if data_dir and data_dir.startswith('gs://'):
        full_path = f"{data_dir}/{dataset_name}/0.1.0"
        builder = tfds.builder_from_directory(builder_dir=full_path)
        ds = builder.as_dataset(split=split, shuffle_files=True)
    else:
        ds = tfds.load(dataset_name, split=split, shuffle_files=True, data_dir=data_dir)
    
    # Take one episode
    for episode in ds.take(1):
        steps = list(episode['steps'])
        
        images = []
        proprio = []
        
        for s in steps:
            # Image
            img = s['observation']['image']
            img = tf.image.resize(img, (128, 128))
            img = tf.cast(img, tf.float32) / 255.0
            images.append(tf.transpose(img, [2, 0, 1]).numpy())
            
            # Proprio (7D)
            obs = s['observation']
            p = np.zeros(6, dtype=np.float32)
            if 'ee_pose' in obs: p = obs['ee_pose'].numpy()
            elif 'pose' in obs: p = obs['pose'].numpy()
            
            g = 0.0
            if 'gripper_closed' in obs: g = obs['gripper_closed'].numpy()
            elif 'gripper_state' in obs: g = obs['gripper_state'].numpy()
            if not np.isscalar(g): g = g.item() if g.size == 1 else g[0]
            
            p7 = np.zeros(7, dtype=np.float32)
            p7[:6] = p[:6] if p.shape[0] >= 6 else np.pad(p, (0, 6-p.shape[0]))
            p7[6] = g
            proprio.append(p7)
            
        return np.array(images), np.array(proprio)
    return None, None

def make_gif(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Configuration (Must match training) ---
    config = {
        'device': device,
        'goal_dim': 64, 
        'cond_dim': 256,
        'vision_feature_dim': 256,
        'num_vision_tokens': 8,
        'window_size': 8,
        'latent_dim': 256,
        'queue_size': 10,
        'embed_dim': 256,
        'num_heads': 4,
        'num_layers': 4,
        'proprio_dim': 7,
        'action_dim': 7
    }

    # --- Load Model ---
    model = AxisModel(config).to(device)
    model.eval()

    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pt')
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Filter state dict
    state_dict = checkpoint['model_state_dict']
    model_state = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered_state_dict, strict=False)
    
    step = checkpoint.get('step', 0)

    # --- Load Data ---
    print("Loading one episode...")
    images_np, proprio_np = load_episode(args.dataset, args.data_dir)
    
    if images_np is None:
        print("Failed to load episode.")
        return

    T = images_np.shape[0]
    print(f"Episode length: {T}")

    # Prepare Inputs
    images = torch.tensor(images_np, dtype=torch.float32).to(device).unsqueeze(0) # (1, T, 3, 128, 128)
    proprio = torch.tensor(proprio_np, dtype=torch.float32).to(device).unsqueeze(0) # (1, T, 7)
    
    # Goal: Use first and last frame to encode goal (Mock Oracle logic)
    # For visualization, we can just use a zero goal or try to use the oracle if available.
    # Let's use a zero goal for simplicity or try to import GoalOracle.
    from training.data.goal_oracle import GoalOracle
    oracle = GoalOracle(output_dim=64)
    
    # Infer task type (simple heuristic)
    start_grip = proprio_np[0][6]
    end_grip = proprio_np[-1][6]
    task_type = 0
    if start_grip < 0.5 and end_grip > 0.5: task_type = 1 # Pick
    elif start_grip > 0.5 and end_grip < 0.5: task_type = 2 # Place
    
    goal_emb = oracle.encode_goal(task_type, proprio_np[0], proprio_np[-1])
    goal_emb = goal_emb.to(device).unsqueeze(0) # (1, 64)

    # --- Inference Loop ---
    pred_actions = []
    requery_preds = []
    
    model.reset_memory(1)
    W = config['window_size']
    
    with torch.no_grad():
        for t in range(T):
            # Construct Window ending at t
            # If t < W, pad with first frame
            if t < W:
                # Pad
                pad_len = W - 1 - t
                # Current window is 0..t (length t+1)
                # Need W. Pad (W - (t+1)) = W - t - 1
                
                curr_imgs = images[:, :t+1]
                curr_props = proprio[:, :t+1]
                
                pad_imgs = curr_imgs[:, 0:1].repeat(1, pad_len, 1, 1, 1)
                pad_props = curr_props[:, 0:1].repeat(1, pad_len, 1)
                
                win_imgs = torch.cat([pad_imgs, curr_imgs], dim=1)
                win_props = torch.cat([pad_props, curr_props], dim=1)
            else:
                # Slice t-W+1 to t+1 (length W)
                win_imgs = images[:, t-W+1 : t+1]
                win_props = proprio[:, t-W+1 : t+1]
            
            # Forward
            # Goal is constant for the episode in this simple viz
            pred_act, _, req_logit = model(
                win_imgs, 
                win_props, 
                goal_emb, 
                update_queue=True, 
                use_memory=True
            )
            
            pred_actions.append(pred_act.cpu())
            requery_preds.append(torch.sigmoid(req_logit).cpu())

    pred_actions = torch.stack(pred_actions, dim=1).squeeze(0) # (T, 7)
    requery_preds = torch.stack(requery_preds, dim=1).squeeze(0) # (T, 1)
    
    # Target actions: Shift proprio by 1?
    # In training, target at step t is usually proprio[t+1].
    # For viz, let's just use proprio[t] as "current state" and maybe proprio[t+1] as target.
    # Or just plot proprio[t] as ground truth action (if action is absolute pose).
    # Since our model predicts next pose, let's compare pred[t] with proprio[t+1].
    
    targets = np.zeros_like(proprio_np)
    targets[:-1] = proprio_np[1:]
    targets[-1] = proprio_np[-1] # Repeat last
    
    targets = torch.tensor(targets, dtype=torch.float32)

    # --- Generate GIF ---
    viz = Visualizer(args.checkpoint_dir)
    viz.create_gif(step, images[0], targets, pred_actions, requery_preds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fractal20220817_data')
    parser.add_argument('--data_dir', type=str, default='gs://gresearch/robotics')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    make_gif(args)
