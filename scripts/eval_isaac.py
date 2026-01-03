
# scripts/eval_isaac.py
import argparse
import os
import sys
import time
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from scipy.spatial.transform import Rotation as R
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt
# Required for 3D plotting
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 0. Bootstrap Isaac Sim (Required for 4.0+)
try:
    import isaacsim
except ImportError:
    pass 

# 1. Path Setup for Isaac Lab (Legacy vs Standard)
cwd = os.getcwd()
possible_extension_paths = [
    os.path.join(cwd, "source"), 
    os.path.join(cwd, "source", "isaaclab")
]
for p in possible_extension_paths:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

# 2. AppLauncher Import using Fallbacks
try:
    from omni.isaac.lab.app import AppLauncher
except ImportError:
    try:
        from isaaclab.app import AppLauncher
    except ImportError as e:
        print(f"CRITICAL ERROR: Failed to import AppLauncher. {e}")
        sys.exit(1)

def main():
    # 3. Parse Args
    parser = argparse.ArgumentParser(description="Evaluate Axis Model in Isaac Lab")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--robot', type=str, default='franka', choices=['franka', 'google'], help="Robot to use")
    parser.add_argument('--steps', type=int, default=1000, help="Number of steps to run")
    parser.add_argument('--video', action='store_true', help="Record video of the evaluation")
    parser.add_argument('--model_refresh', type=int, default=60, help="Hz for model updates")
    
    # AppLauncher Args
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # 4. Launch App
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # 5. Imports
    import torch
    
    try:
        from src.models.axis_v1 import AxisModel
    except ImportError as e:
         print(f"ERROR: Could not import AxisModel. {e}")
         simulation_app.close()
         sys.exit(1)

    try:
        from my_robot_ext.tasks.eval_env import AxisEvalEnv, AxisEvalEnvCfg
        from my_robot_ext.wrappers.axis_wrapper import AxisObservationWrapper
        from my_robot_ext.config.robots import FrankaCfg, GoogleRobotCfg
    except ImportError as e:
        print(f"ERROR: Failed to import my_robot_ext: {e}")
        simulation_app.close()
        sys.exit(1)
        
    try:
        from imitation.data.goal_oracle import GoalOracle
    except ImportError:
        print("ERROR: Could not import GoalOracle.")
        simulation_app.close()
        sys.exit(1)

    # 6. Run Evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading model from {args.checkpoint}...")
    config = {
        'device': device,
        'goal_dim': 64, 
        'embed_dim': 256,
        'window_size': 8,
    }
    
    try:
        model = AxisModel(config).to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            keys_to_remove = [k for k in state_dict.keys() if 'latent_queue.queue' in k]
            for k in keys_to_remove:
                del state_dict[k]
        
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        print("Model loaded successfully.")
        
        oracle = GoalOracle(output_dim=64)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        simulation_app.close()
        return

    # Setup Environment
    try:
        env_cfg = AxisEvalEnvCfg()
        if args.robot == 'franka': env_cfg.robot = FrankaCfg()
        elif args.robot == 'google': env_cfg.robot = GoogleRobotCfg()
        
        render_mode = "rgb_array" if args.video else None
        env = AxisEvalEnv(cfg=env_cfg, render_mode=render_mode)
        
        if args.video:
            video_folder = os.path.join(os.path.dirname(args.checkpoint), "videos")
            
            class CameraRenderWrapper(gym.Wrapper):
                def render(self):
                    try:
                        cam_sensor = self.unwrapped.scene["camera"]
                        rgb = cam_sensor.data.output["rgb"][0] # (H, W, 4)
                        img = rgb.cpu().numpy()
                        if img.shape[-1] == 4: img = img[..., :3]
                        if img.dtype != np.uint8:
                            if img.max() <= 1.05: img = (img * 255).astype(np.uint8)
                            else: img = img.astype(np.uint8)
                        return img
                    except:
                        return np.zeros((128, 128, 3), dtype=np.uint8)

            env = CameraRenderWrapper(env)
            env = RecordVideo(env, video_folder=video_folder, step_trigger=lambda step: step == 0, video_length=args.steps, name_prefix="eval_run")
            
        env = AxisObservationWrapper(env, window_size=8, device=device)
    except Exception as e:
        print(f"Error initializing environment: {e}")
        simulation_app.close()
        return

    # Loop
    try:
        obs, info = env.reset()
        
        B = env.num_envs
        goal_emb = torch.zeros(B, 64, device=device)
        target_dt = 1.0 / args.model_refresh
        last_command_time = 0
        requery = 0.0
        
        initial_pose_np = None
        target_quat = None

        # --- STUCK DETECTION ---
        stuck_timer = time.time()
        last_stuck_pos = None
        STUCK_THRESHOLD = 0.01 # 1cm
        STUCK_TIME = 3.0       # 3 seconds
        # -----------------------

        history_actual, history_pred, history_target = [], [], []
        print("Starting Simulation Loop (Model Control)...")
        
        env_actions = torch.zeros((B, 8), device=device)
        pos_pred = np.zeros(3)
        
        for i in range(args.steps):
            with torch.no_grad():
                images = obs['images'] # (B, W, C, H, W)
                proprio = obs['proprio'] # (B, W, 8)
                
                if i == 0:
                    print("\n[Input Validation]")
                    assert images.shape[1] == 8
                    assert images.shape[-2:] == (128, 128)
                    print("Input shapes validated: OK")
                    print(f"INIT Proprio (Mirrored Y Tip): {proprio[0, -1].cpu().numpy()}")

                # --- Goal Logic ---
                if initial_pose_np is None:
                    initial_pose_np = proprio[0, -1, :7].cpu().numpy()
                    target_quat = initial_pose_np[3:] 
                    print("Goal Logic: Latched Start Pose & Target Quat")

                # Get Cube Pos (Sim Frame)
                if 'cube' in env.unwrapped.scene.keys():
                    cube_pos = env.unwrapped.scene['cube'].data.root_pos_w[:, :3]
                    cube_pos_np = cube_pos[0].cpu().numpy()
                else:
                    cube_pos_np = np.zeros(3)

                # --- VISUALIZATION CALCS ---
                # Proprio is Tip Frame (Mirrored Y)
                pos_tip_mirrored = proprio[0, -1, :3].cpu().numpy()
                quat_xyzw = proprio[0, -1, 3:7].cpu().numpy()
                
                # UN-MIRROR for plotting (Sim Frame)
                pos_tip_sim = pos_tip_mirrored.copy()
                pos_tip_sim[1] *= -1.0
                
                # Append TIP for Plotting (Sim Frame)
                history_actual.append(pos_tip_sim)
                
                # Command Plotting:
                # Command is Model Frame (Mirrored Y).
                # Flip to visualize in Sim Frame.
                pos_pred_sim = pos_pred.copy()
                pos_pred_sim[1] *= -1.0
                history_pred.append(pos_pred_sim)
                
                history_target.append(cube_pos_np)

                # --- STUCK CHECK ---
                stuck_trigger = False
                current_time_sim = time.time()
                
                if last_stuck_pos is None:
                     last_stuck_pos = pos_tip_sim
                     stuck_timer = current_time_sim
                else:
                    dist = np.linalg.norm(pos_tip_sim - last_stuck_pos)
                    if dist > STUCK_THRESHOLD:
                        last_stuck_pos = pos_tip_sim
                        stuck_timer = current_time_sim
                    elif (current_time_sim - stuck_timer) > STUCK_TIME:
                        print(f"[Warn] Stuck Detected! (No mov > {STUCK_THRESHOLD}m for {STUCK_TIME}s). Requerying...", flush=True)
                        stuck_trigger = True
                        stuck_timer = current_time_sim 
                # -------------------

                # Construct Goal
                end_p = np.zeros(7, dtype=np.float32)
                end_p[:3] = cube_pos_np
                
                # --- GOAL Y INVERSION ---
                # To match Mirrored Proprio.
                end_p[1] *= -1.0 
                # ------------------------
                
                # Force Downward Orientation for Goal
                end_p[3:] = np.array([0.0, 0.0, 1.0, 0.0]) 
                
                if i == 0 or float(requery) > 0.6 or stuck_trigger:
                    # Update Start Pose to Current Pose (Tip Frame) for Requery
                    initial_pose_np = proprio[0, -1, :7].cpu().numpy()

                    task_type = 1 
                    g_emb = oracle.encode_goal(task_type, initial_pose_np, end_p).to(device)
                    goal_emb[0] = g_emb
                    print(f"Goal Re-generated! Start: {initial_pose_np[:3].round(3)}", flush=True)

                # Model Forward Pass
                current_time = time.time()
                model_ran = False
                if current_time - last_command_time >= target_dt:
                    t0 = time.time()
                    with torch.inference_mode():
                        actions, _, requery_logit = model(images, proprio, goal_embedding=goal_emb, update_queue=False, use_memory=False)
                        requery = torch.sigmoid(requery_logit)
                    dt = (time.time() - t0) * 1000
                    last_command_time = current_time
                    pos_pred = actions[0, :3].cpu().numpy()
                    model_ran = True

                # --- LOGGING ---
                if i % 10 == 0:
                    delta = pos_tip_sim - cube_pos_np
                    print(f"S:{i} | Tip:{pos_tip_sim.round(3)} | Cube:{cube_pos_np.round(3)} | D:{np.linalg.norm(delta):.3f} | dZ:{delta[2]:.3f}")
                
                 # Step
                
                if model_ran:
                    act_pos = actions[:, :3]
                    act_quat_xyzw = actions[:, 3:7]
                    act_grip = actions[:, 7:]
                    
                    # --- ACTION INVERSION ---
                    # Model commands Mirrored Y. Flip to Sim Y.
                    act_pos_sim = act_pos.clone()
                    act_pos_sim[:, 1] *= -1.0 
                    # ------------------------
                    
                    act_quat_wxyz = torch.cat([act_quat_xyzw[:, 3:4], act_quat_xyzw[:, :3]], dim=1)
                    
                    act_grip_prob = torch.sigmoid(act_grip)
                    act_grip_cmd = torch.where(act_grip_prob > 0.5, 
                                            torch.tensor(1.0, device=device), 
                                            torch.tensor(-1.0, device=device))
                    
                    env_actions = torch.cat([act_pos_sim, act_quat_wxyz, act_grip_cmd], dim=1)
                
                obs, reward, terminated, truncated, info = env.step(env_actions)
                        
                if terminated.any() or truncated.any():
                    print(f"!!! RESET triggered at step {i} !!!")
                    obs, info = env.reset()
                    initial_pose_np = None 
                    
        # Plotting
        print("Generating 3D Trajectory Plot...")
        traj_actual = np.array(history_actual)
        traj_pred = np.array(history_pred)
        traj_target = np.array(history_target)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(traj_actual[:, 0], traj_actual[:, 1], traj_actual[:, 2], label='Actual TIP Path', color='blue')
        ax.scatter(traj_actual[0,0], traj_actual[0,1], traj_actual[0,2], color='blue', marker='o')
        
        # Plot commands in Red. Note these are PRE-FLIP (Model Output).
        ax.plot(traj_pred[:, 0], traj_pred[:, 1], traj_pred[:, 2], label='Model Commands (Flipped)', color='red', linestyle='--')
        
        ax.plot(traj_target[:, 0], traj_target[:, 1], traj_target[:, 2], label='Cube Path', color='green', alpha=0.6)
        ax.scatter(traj_target[-1,0], traj_target[-1,1], traj_target[-1,2], color='green', marker='*', s=200, label='Cube End')

        ax.set_title(f'Sim-to-Real Debug: Trajectory Analysis\nSteps: {len(traj_actual)}')
        ax.legend()
        plt.savefig(os.path.join(os.path.dirname(args.checkpoint), "debug_trajectory_3d.png"))
        plt.close()
        
        print("Evaluation Complete.")
    except Exception as e:
        print(f"Runtime error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals(): env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()
