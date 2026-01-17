import os
import sys
import torch
import argparse
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from scipy.spatial.transform import Rotation as R

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.axis import AxisModel
from imitation.utils.visualizer import Visualizer
from imitation.data.goal_oracle import GoalOracle
from src.utils.rotation_utils import quaternion_to_rotation_6d

def load_episode(dataset_name, data_dir, split='train'):
    """Loads a single random episode and converts to 10D pose format."""
    print(f"Loading dataset {dataset_name} from {data_dir}...")
    
    # Special handling for fractal on GCS
    if data_dir and data_dir.startswith('gs://') and 'fractal' in dataset_name:
        full_path = f"{data_dir}/{dataset_name}/0.1.0"
        builder = tfds.builder_from_directory(builder_dir=full_path)
        ds = builder.as_dataset(split=split, shuffle_files=True)
    else:
        # Standard load
        ds = tfds.load(dataset_name, split=split, shuffle_files=True, data_dir=data_dir)
    
    # Use safe iteration
    iterator = iter(ds)
    episode = next(iterator)
    steps = list(episode['steps'])
    
    images = []
    proprio = []
    
    image_keys = ['image', 'exterior_image_1_left', 'wrist_image_left', 'exterior_image_2_left']
    
    for s in steps:
        obs = s['observation']
        
        # Image Processing
        img = None
        for key in image_keys:
            if key in obs:
                img = obs[key]
                break
        
        if img is not None:
            img = tf.image.resize(img, (128, 128))
            img = tf.cast(img, tf.float32) / 255.0
            images.append(tf.transpose(img, [2, 0, 1]).numpy())
        else:
            images.append(np.zeros((3, 128, 128), dtype=np.float32))
        
        # Identity quat default
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        pos = np.zeros(3, dtype=np.float32)

        # Pose Extraction
        if 'base_pose_tool_reached' in obs:
            p = obs['base_pose_tool_reached'].numpy() # 7D: [x,y,z,qx,qy,qz,qw]
            pos = p[:3]
            quat = p[3:7]
        elif 'cartesian_position' in obs:
            p6 = obs['cartesian_position'].numpy() # 6D: [x,y,z,rx,ry,rz]
            if p6.shape[0] == 6:
                pos = p6[:3]
                euler = p6[3:]
                quat = R.from_euler('xyz', euler).as_quat()
        elif 'ee_pose' in obs:
            p = obs['ee_pose'].numpy()
            if p.shape[0] >= 7:
                pos = p[:3]
                quat = p[3:7]
        elif 'pose' in obs:
            p = obs['pose'].numpy()
            if p.shape[0] >= 7:
                pos = p[:3]
                quat = p[3:7]
        
        # Gripper Extraction
        g = 0.0
        if 'gripper_closed' in obs: g = obs['gripper_closed'].numpy()
        elif 'gripper_position' in obs: g = obs['gripper_position'].numpy()
        elif 'gripper_state' in obs: g = obs['gripper_state'].numpy()
        
        if not np.isscalar(g): g = g.item() if g.size == 1 else g[0]
        
        # Normalize Quat
        qn = np.linalg.norm(quat)
        if qn < 1e-8: quat = np.array([0., 0., 0., 1.], dtype=np.float32)
        else: quat = quat / qn
        
        # Convert to 10D: [Pos(3) + Rot6D(6) + Gripper(1)]
        rot_6d = quaternion_to_rotation_6d(torch.tensor(quat, dtype=torch.float32)).numpy()
        p10 = np.zeros(10, dtype=np.float32)
        p10[:3] = pos
        p10[3:9] = rot_6d
        p10[9] = g
        proprio.append(p10)
        
    return np.array(images), np.array(proprio)

def make_gif(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- V2 Configuration ---
    config = {
        'device': device,
        'goal_dim': 64, 
        'embed_dim': 256,
        'num_heads': 4,
        'num_layers': 4,
        'proprio_dim': 10, # 10D Pose
        'action_dim': 10,  # 10D Action
        'use_rope': True
    }

    # --- Load Model ---
    model = AxisModel(config).to(device)
    model.eval()

    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pt')
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        files = [f for f in os.listdir(args.checkpoint_dir) if f.startswith('run_') and f.endswith('.pt')]
        if files:
            files.sort(key=lambda x: os.path.getmtime(os.path.join(args.checkpoint_dir, x)))
            checkpoint_path = os.path.join(args.checkpoint_dir, files[-1])
            print(f"Using latest run checkpoint: {checkpoint_path}")
        else:
            return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both full checkpoint and simple state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model_state = model.state_dict()
    
    # Filter/Migrate logic (if needed, but assuming fresh V2 training here)
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
    proprio = torch.tensor(proprio_np, dtype=torch.float32).to(device).unsqueeze(0) # (1, T, 10)
    
    # --- Goal Oracle (V2) ---
    oracle = GoalOracle(output_dim=64)
    
    # Infer task type
    start_grip = proprio_np[0][9] # Gripper is index 9 in 10D pose
    end_grip = proprio_np[-1][9]
    task_type = 0
    if start_grip < 0.5 and end_grip > 0.5: task_type = 1 # Pick
    elif start_grip > 0.5 and end_grip < 0.5: task_type = 2 # Place
    
    print(f"Inferred Task Type: {task_type} (Start Grip: {start_grip:.2f}, End Grip: {end_grip:.2f})")
    
    # Goal needs start/end poses
    goal_emb = oracle.encode_goal(task_type, proprio_np[0], proprio_np[-1])
    goal_emb = goal_emb.to(device).unsqueeze(0) # (1, 64)

    # --- Inference Loop ---
    pred_actions = []
    requery_preds = []
    
    W = 8 # Window size
    
    import time
    print("Running inference (Autoregressive V2)...")
    inference_times = []
    
    # Clone proprio for autoregressive updates (starts as Ground Truth)
    simulated_proprio = proprio.clone()
    
    with torch.no_grad():
        for t in range(T):
            if t % 10 == 0: print(f"Step {t}/{T}")
            
            # Construct Window
            if t < W:
                pad_len = W - 1 - t
                curr_imgs = images[:, :t+1]
                curr_props = simulated_proprio[:, :t+1]
                pad_imgs = curr_imgs[:, 0:1].repeat(1, pad_len, 1, 1, 1)
                pad_props = curr_props[:, 0:1].repeat(1, pad_len, 1)
                win_imgs = torch.cat([pad_imgs, curr_imgs], dim=1)
                win_props = torch.cat([pad_props, curr_props], dim=1)
            else:
                win_imgs = images[:, t-W+1 : t+1]
                win_props = simulated_proprio[:, t-W+1 : t+1]
            
            # Forward Pass
            start_time = time.time()
            # V2 forward returns (action_delta, requery)
            action_delta, req_logit = model(win_imgs, win_props, goal_emb)
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            # Update Simulation State (Apply Delta)
            # action_delta is (1, 10)
            current_pose = simulated_proprio[:, t, :] # (1, 10)
            next_pose = current_pose + action_delta
            
            # Store predictions
            pred_actions.append(action_delta.cpu())
            requery_preds.append(torch.sigmoid(req_logit).cpu())
            
            # Update next simulated input if available
            if t + 1 < T:
                simulated_proprio[:, t+1, :] = next_pose

    pred_actions = torch.stack(pred_actions, dim=1).squeeze(0) # (T, 10)
    requery_preds = torch.stack(requery_preds, dim=1).squeeze(0) # (T, 1)
    
    # Target Delta Actions (ShiftGT - GT)
    # T steps, T-1 transitions + last static
    # targets should be deltas for V2
    proprio_gt = torch.tensor(proprio_np, dtype=torch.float32)
    targets_delta = np.zeros_like(proprio_np)
    
    # Delta = Next - Curr
    targets_delta[:-1] = proprio_np[1:] - proprio_np[:-1]
    targets_delta[-1] = 0 # Last step delta is 0
    targets = torch.tensor(targets_delta, dtype=torch.float32)

    # --- Generate GIF ---
    print("Generating GIF...")
    viz = Visualizer(args.checkpoint_dir)
    viz.create_gif(
        step, 
        images[0], 
        targets, 
        pred_actions, 
        requery_preds, 
        inference_times=inference_times, 
        save_prefix='manual_episode_v2', 
        dpi=100
    )
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fractal20220817_data')
    parser.add_argument('--data_dir', type=str, default='gs://gresearch/robotics')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    make_gif(args)
