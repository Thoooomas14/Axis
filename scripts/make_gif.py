import os
import sys
import torch
import argparse
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.model.axis_v1 import AxisModel
from training.utils.visualizer import Visualizer
from training.data.goal_oracle import GoalOracle

def load_episode(dataset_name, data_dir, split='train'):
    """Loads a single random episode from the dataset safely."""
    print(f"Loading dataset {dataset_name} from {data_dir}...")
    if data_dir and data_dir.startswith('gs://'):
        full_path = f"{data_dir}/{dataset_name}/0.1.0"
        builder = tfds.builder_from_directory(builder_dir=full_path)
        ds = builder.as_dataset(split=split, shuffle_files=True)
    else:
        ds = tfds.load(dataset_name, split=split, shuffle_files=True, data_dir=data_dir)
    
    # Use safe iteration
    iterator = iter(ds)
    episode = next(iterator)
    
    steps = list(episode['steps'])
    images = []
    proprio = []
    
    for s in steps:
        # Image
        img = s['observation']['image']
        img = tf.image.resize(img, (128, 128))
        img = tf.cast(img, tf.float32) / 255.0
        images.append(tf.transpose(img, [2, 0, 1]).numpy())
        
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
        proprio.append(p8)
        
    return np.array(images), np.array(proprio)

def make_gif(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Configuration ---
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
        'proprio_dim': 8, # 3 Pos + 4 Quat + 1 Gripper
        'action_dim': 8
    }

    # --- Load Model ---
    model = AxisModel(config).to(device)
    model.eval()

    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pt')
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        # Try looking for run_*.pt files
        files = [f for f in os.listdir(args.checkpoint_dir) if f.startswith('run_') and f.endswith('.pt')]
        if files:
            files.sort(key=lambda x: os.path.getmtime(os.path.join(args.checkpoint_dir, x)))
            checkpoint_path = os.path.join(args.checkpoint_dir, files[-1])
            print(f"Using latest run checkpoint: {checkpoint_path}")
        else:
            return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Filter state dict
    state_dict = checkpoint['model_state_dict']
    model_state = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered_state_dict, strict=False)
    
    step = checkpoint.get('step', 0)
    print(f"Loaded model at step {step}")

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
    proprio = torch.tensor(proprio_np, dtype=torch.float32).to(device).unsqueeze(0) # (1, T, 8)
    
    # --- Goal Oracle ---
    oracle = GoalOracle(output_dim=64)
    
    # Infer task type (simple heuristic)
    start_grip = proprio_np[0][7] # Index 7 is gripper
    end_grip = proprio_np[-1][7]
    task_type = 0
    if start_grip < 0.5 and end_grip > 0.5: task_type = 1 # Pick
    elif start_grip > 0.5 and end_grip < 0.5: task_type = 2 # Place
    
    print(f"Inferred Task Type: {task_type} (Start Grip: {start_grip:.2f}, End Grip: {end_grip:.2f})")
    
    goal_emb = oracle.encode_goal(task_type, proprio_np[0], proprio_np[-1])
    goal_emb = goal_emb.to(device).unsqueeze(0) # (1, 64)

    # --- Inference Loop ---
    pred_actions = []
    requery_preds = []
    
    model.reset_memory(1)
    W = config['window_size']
    
    import time
    
    print("Running inference...")
    inference_times = []
    with torch.no_grad():
        for t in range(T):
            if t % 10 == 0: print(f"Step {t}/{T}")
            # Construct Window ending at t
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
            
            # Measure inference time
            start_time = time.time()
            pred_act, _, req_logit = model(
                win_imgs, 
                win_props, 
                goal_emb, 
                update_queue=True, 
                use_memory=True
            )
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            pred_actions.append(pred_act.cpu())
            requery_preds.append(torch.sigmoid(req_logit).cpu())

    pred_actions = torch.stack(pred_actions, dim=1).squeeze(0) # (T, 8)
    requery_preds = torch.stack(requery_preds, dim=1).squeeze(0) # (T, 1)
    
    # Target actions: Shift proprio by 1
    targets = np.zeros_like(proprio_np)
    targets[:-1] = proprio_np[1:]
    targets[-1] = proprio_np[-1]
    targets = torch.tensor(targets, dtype=torch.float32)

    # --- Generate GIF ---
    print("Generating GIF...")
    viz = Visualizer(args.checkpoint_dir)
    # Use 'manual' prefix to distinguish from training GIFs
    # Use higher DPI for manual generation
    viz.create_gif(step, images[0], targets, pred_actions, requery_preds, inference_times=inference_times, save_prefix='manual_episode', dpi=100)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fractal20220817_data')
    parser.add_argument('--data_dir', type=str, default='gs://gresearch/robotics')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    make_gif(args)
