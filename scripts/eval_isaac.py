
# scripts/eval_isaac.py
import argparse
import os
import sys
import time
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 0. Bootstrap Isaac Sim (Required for 4.0+)
try:
    import isaacsim
except ImportError:
    pass 

# 1. Path Setup for Isaac Lab
cwd = os.getcwd()
possible_extension_paths = [
    os.path.join(cwd, "source"), 
    os.path.join(cwd, "source", "isaaclab")
]
for p in possible_extension_paths:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

try:
    from omni.isaac.lab.app import AppLauncher
except ImportError:
    try:
        from isaaclab.app import AppLauncher
    except ImportError as e:
        print(f"CRITICAL ERROR: Failed to import AppLauncher. {e}")
        sys.exit(1)

# Import V2 Components
from src.inference import AxisInference
from src.utils.rotation_utils import quaternion_to_rotation_6d
from imitation.data.goal_oracle import GoalOracle

def quaternion_to_rot6d_numpy(quat_xyzw):
    """Helper to convert numpy quat [x,y,z,w] to 6D."""
    # Using torch utility
    q_tensor = torch.tensor(quat_xyzw, dtype=torch.float32)
    r6 = quaternion_to_rotation_6d(q_tensor).numpy()
    return r6

def main():
    parser = argparse.ArgumentParser(description="Evaluate Axis V2 in Isaac Lab")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--robot', type=str, default='franka', choices=['franka', 'google'], help="Robot type")
    parser.add_argument('--steps', type=int, default=1000, help="Max steps")
    parser.add_argument('--video', action='store_true', help="Record video")
    parser.add_argument('--model_refresh', type=int, default=30, help="Control frequency Hz")
    parser.add_argument('--chunk_size', type=int, default=1, help="Action chunk size used in model")
    
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Launch App
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        from my_robot_ext.tasks.eval_env import AxisEvalEnv, AxisEvalEnvCfg
        from my_robot_ext.wrappers.axis_wrapper import AxisObservationWrapper
        from my_robot_ext.config.robots import FrankaCfg, GoogleRobotCfg
    except ImportError as e:
        print(f"ERROR: Failed to import my_robot_ext: {e}")
        simulation_app.close()
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Load Axis V2 Inference ---
    print(f"Loading AxisInference from {args.checkpoint}...")
    config = {
        'device': device,
        'proprio_dim': 10,
        'action_dim': 7, # Twist=7 (used internally by model)
        'embed_dim': 256,
        'window_size': 8,
        'chunk_size': args.chunk_size, # Passed to inference wrapper
        'max_pos_delta': 0.05,
        'max_rot_delta': 0.1
    }
    
    try:
        agent = AxisInference(args.checkpoint, config, device=device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load agent: {e}")
        simulation_app.close()
        sys.exit(1)

    oracle = GoalOracle(output_dim=64)

    # --- Setup Environment ---
    try:
        env_cfg = AxisEvalEnvCfg()
        if args.robot == 'franka': env_cfg.robot = FrankaCfg()
        elif args.robot == 'google': env_cfg.robot = GoogleRobotCfg()
        
        render_mode = "rgb_array" if args.video else None
        env = AxisEvalEnv(cfg=env_cfg, render_mode=render_mode)
        
        if args.video:
            video_folder = os.path.join(os.path.dirname(args.checkpoint), "videos")
            env = RecordVideo(env, video_folder=video_folder, step_trigger=lambda step: step == 0, video_length=args.steps, name_prefix="eval_v2")
            
        # Wrapper now returns 10D proprio
        env = AxisObservationWrapper(env, window_size=8, device=device)
    except Exception as e:
        print(f"Error initializing environment: {e}")
        simulation_app.close()
        return

    # --- Evaluation Loop ---
    try:
        obs, info = env.reset()
        agent.reset() # Reset inference state (cache/ensembler)
        
        B = env.num_envs
        
        # State
        initial_pose_10d = None
        target_pos_sim = None 
        goal_vector = np.zeros((B, 64), dtype=np.float32)
        requery_flag = True
        
        print("Starting Control Loop...")
        
        for i in range(args.steps):
            # 1. Get Observations (Torch -> Numpy)
            # Images: (B, W, C, H, W)
            images_np = obs['images'].cpu().numpy()
            # Proprio: (B, W, 10) [pos3, rot6d, grip1] (already handled by wrapper)
            proprio_10d_np = obs['proprio'].cpu().numpy()
            
            # Use last proprio in window as current state
            current_pose_10d = proprio_10d_np[0, -1] # (10,)
            
            # 3. Goal Logic (Sim specific hacking)
            if 'cube' in env.unwrapped.scene.keys():
                cube_pos = env.unwrapped.scene['cube'].data.root_pos_w[0, :3].cpu().numpy()
            else:
                cube_pos = np.zeros(3)
                
            if initial_pose_10d is None or requery_flag:
                initial_pose_10d = current_pose_10d.copy()
                
                # Goal Target: Cube Pos (converted to Model Frame)
                target_pos_model = cube_pos.copy()
                target_pos_model[1] *= -1.0 # Mirror Y
                
                # Goal Pose: Target Pos + Fixed Downward Orientation
                target_pose_10d = np.zeros(10, dtype=np.float32)
                target_pose_10d[:3] = target_pos_model
                
                # Fixed Rotation: Downward [0,0,1,0] xyzw -> 6D
                target_quat_xyzw = np.array([0.0, 0.0, 1.0, 0.0])
                target_rot6d = quaternion_to_rot6d_numpy(target_quat_xyzw)
                target_pose_10d[3:9] = target_rot6d
                target_pose_10d[9] = 1.0 # Close gripper
                
                # Encode Goal
                # Task 1: Pick
                g_emb = oracle.encode_goal(1, initial_pose_10d, target_pose_10d)
                goal_vector[0] = g_emb.numpy()
                
                print(f"Goal Updated! Target: {target_pos_model.round(3)}")
                requery_flag = False

            # 4. Predict
            # Call agent with single sample (remove batch dim)
            batch_result = agent.predict(
                images_np[0], # (W, C, H, W)
                proprio_10d_np[0], # (W, 10)
                goal_vector[0] # (64,)
            )
            
            requery_flag = batch_result['requery']
            
            # 5. Execute
            # Model Output: 'action_absolute' is 10D (pos3, rot6d, grip1)
            act_pos = batch_result['position']
            act_quat_xyzw = batch_result['quaternion']
            act_grip = batch_result['gripper']
            
            # Flip Y for Sim (Model -> Sim)
            act_pos_sim = act_pos.copy()
            act_pos_sim[1] *= -1.0
            
            # Quat conversion xyzw -> wxyz for Isaac
            act_quat_wxyz = np.array([act_quat_xyzw[3], act_quat_xyzw[0], act_quat_xyzw[1], act_quat_xyzw[2]])
            
            # Gripper Cmd
            act_grip_cmd = 1.0 if act_grip > 0.5 else -1.0
            
            # Compose
            env_action = np.concatenate([act_pos_sim, act_quat_wxyz, [act_grip_cmd]])
            env_action_t = torch.tensor(env_action, device=device).unsqueeze(0) # (1, 8)
            
            # Step
            obs, reward, terminated, truncated, info = env.step(env_action_t)
            
            if i % 10 == 0:
                print(f"Step {i} | Pos:{current_pose_10d[:3].round(3)} | Grip:{current_pose_10d[9]:.2f}")
                
            if terminated.any() or truncated.any():
                print("Reset triggered.")
                obs, info = env.reset()
                agent.reset() # Reset inference state
                initial_pose_10d = None
                requery_flag = True

        print("Evaluation Complete.")
        
    except Exception as e:
        print(f"Runtime Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
