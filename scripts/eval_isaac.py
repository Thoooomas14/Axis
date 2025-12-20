
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

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 0. Bootstrap Isaac Sim (Required for 4.0+)
try:
    import isaacsim
except ImportError:
    pass 

# 1. Path Setup for Isaac Lab (Legacy vs Standard)
# Try to detect if we need to add source/ directories to path
# This is common in dev installs of Isaac Lab
cwd = os.getcwd()
possible_extension_paths = [
    os.path.join(cwd, "source"), 
    os.path.join(cwd, "source", "isaaclab")
]
for p in possible_extension_paths:
    if os.path.exists(p) and p not in sys.path:
        print(f"[eval_isaac] Appending to sys.path: {p}")
        sys.path.append(p)

# 2. AppLauncher Import using Fallbacks
try:
    from omni.isaac.lab.app import AppLauncher
except ImportError:
    try:
        from isaaclab.app import AppLauncher
        print("[eval_isaac] Using 'isaaclab' namespace.")
    except ImportError as e:
        print(f"CRITICAL ERROR: Failed to import AppLauncher (checked omni.isaac.lab and isaaclab). {e}")
        print("Ensure you are running this script within the Isaac Sim python environment.")
        print("Usage: .\\isaaclab.bat -p path\\to\\eval_isaac.py ...")
        sys.exit(1)

def main():
    # 3. Parse Args
    parser = argparse.ArgumentParser(description="Evaluate Axis Model in Isaac Lab")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--robot', type=str, default='franka', choices=['franka', 'google'], help="Robot to use")

    parser.add_argument('--steps', type=int, default=1000, help="Number of steps to run")
    parser.add_argument('--video', action='store_true', help="Record video of the evaluation")
    
    # AppLauncher Args
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # 4. Launch App
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # 5. Imports that depend on the App being launched
    import torch
    
    # Attempt to import Axis Model
    try:
        from src.models.axis_v1 import AxisModel
    except ImportError as e:
         print(f"ERROR: Could not import AxisModel. check PYTHONPATH. {e}")
         simulation_app.close()
         sys.exit(1)

    # Attempt to import Extension
    try:
        # We need to import the Configs/Env. 
        # These now have fallback logic inside them, so we just import normally.
        from my_robot_ext.tasks.eval_env import AxisEvalEnv, AxisEvalEnvCfg
        from my_robot_ext.wrappers.axis_wrapper import AxisObservationWrapper
        from my_robot_ext.config.robots import FrankaCfg, GoogleRobotCfg
    except ImportError as e:
        print(f"ERROR: Failed to import my_robot_ext: {e}")
        print("Did you install the extension? Run: pip install -e isaac_sim")
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
        
        # FIX: Remove latent_queue from state_dict if present.
        # The checkpoint has a queue size matching the training batch (e.g. 22),
        # but evaluation runs with batch size 1 (or num_envs).
        # We start with a fresh memory anyway.
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            keys_to_remove = [k for k in state_dict.keys() if 'latent_queue.queue' in k]
            for k in keys_to_remove:
                print(f"Removing {k} from checkpoint (shape mismatch expected).")
                del state_dict[k]
        
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        print("Model loaded successfully.")
    
        # Initialize Goal Oracle Logic Inline
        # MATCH TRAINING EXACTLY:
        # Training (goal_oracle.py) inits (46, 64) matrix and uses slice.
        # We must generate (46, 64) to get the same random sequence.
        rng = np.random.RandomState(42)
        full_proj = rng.randn(46, 64).astype(np.float32) # 7+7+32 = 46
        proj_matrix = full_proj[:17] # Slice first 17 rows for our input
        print("Initialized Goal Projection Matrix (Matched Training: 46->17x64)")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        simulation_app.close()
        return

    # Setup Environment
    print(f"Setting up Isaac Lab Environment for {args.robot}...")
    try:
        env_cfg = AxisEvalEnvCfg()
        
        if args.robot == 'franka':
            env_cfg.robot = FrankaCfg()
        elif args.robot == 'google':
            env_cfg.robot = GoogleRobotCfg()
        
        # Create Env
        render_mode = "rgb_array" if args.video else None
        env = AxisEvalEnv(cfg=env_cfg, render_mode=render_mode)
        
        # Wrap for Video
        if args.video:
            print("Enabling Video Recording -> videos/eval_run.mp4")
            video_folder = os.path.join(os.path.dirname(args.checkpoint), "videos")
            env = RecordVideo(
                env, 
                video_folder=video_folder,
                step_trigger=lambda step: step == 0, # Record from start
                video_length=args.steps, # Record full length
                name_prefix="eval_run"
            )
            
        # Wrap Env
        env = AxisObservationWrapper(env, window_size=8, device=device)
    except Exception as e:
        print(f"Error initializing environment: {e}")
        simulation_app.close()
        return

    # Loop
    try:
        obs, info = env.reset()
        
        # --- DEBUG: Inspect Keys ---
        print(f"[Main] Reset Obs Keys: {list(obs.keys())}")
        if 'policy' in obs:
             print(f"[Main] Policy Keys: {list(obs['policy'].keys())}")
        # ---------------------------
        
        B = env.num_envs
        # Initialize goal with zeros (or zeros then filled)
        goal_emb = torch.zeros(B, 64, device=device)
        
        print("Starting Simulation Loop...")
        
        for i in range(args.steps):
            with torch.no_grad():
                images = obs['images'] # (B, W, C, H, W)
                proprio = obs['proprio']
                
                
                # --- Update Goal using GoalOracle ---
                # Get cube pos (End Pose)
                if 'cube' in env.unwrapped.scene.keys():
                    cube_pos = env.unwrapped.scene['cube'].data.root_pos_w[:, :3] # (B, 3)
                    # Convert to numpy for Oracle
                    cube_pos_np = cube_pos[0].cpu().numpy()
                else:
                    cube_pos_np = np.zeros(3)

                # Get Current Pose (Start Pose) - Use last proprio
                # Proprio is 8D (Pos[3], Quat[4], Grip[1])? 
                # Wrapper might output 8D.
                # Let's assume proprio matches format needed.
                current_pose_np = proprio[0, -1, :7].cpu().numpy() # Take 7D pose
                
                # Encode Goal Inline
                # 1. Type (Pick=1) -> One Hot [0, 1, 0]
                type_vec = np.zeros(3, dtype=np.float32)
                type_vec[1] = 1.0
                
                # 2. Concat [Type, Start, End]
                # Ensure 7D
                start_p = current_pose_np[:7]
                
                # End Pose: Cube Pos + Downward Quat
                # Cube Pos is (3,). We need 7D.
                # Proprio is Scalar Last (xyzw).
                # Standard Downward Grasp: x=1, y=0, z=0, w=0 (180 deg rot around X)
                # Or just [0, 1, 0, 0]? Let's use [1, 0, 0, 0] (x=1) as placeholder or copy current quat?
                # Better: Use the CURRENT orientation as the target orientation? 
                # No, we want to pick it up. Hand likely needs to rotate.
                # Let's use [1, 0, 0, 0] (x,y,z,w) as a valid non-zero quat.
                default_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                end_p = np.concatenate([cube_pos_np, default_quat])
                
                input_vec = np.concatenate([type_vec, start_p, end_p]) # Should be 17
                
                # 3. Project
                # proj_matrix is (17, 64)
                g_emb_np = np.dot(input_vec, proj_matrix)
                
                # 4. Normalize
                g_emb_np = g_emb_np / (np.linalg.norm(g_emb_np) + 1e-6)
                
                goal_emb[0] = torch.tensor(g_emb_np, device=device, dtype=torch.float32)
                
                # Debug Target for log
                tgt = cube_pos_np.tolist()
                # ----------------------------------

                # --- DEBUG: Log Inputs ---
                if i % 10 == 0:
                    img_mean = images.float().mean().item()
                    img_max = images.float().max().item()
                    prop_mean = proprio.float().mean().item()
                    goal_mean = goal_emb.float().mean().item()
                    
                    print(f"  > Inputs[{i}]:")
                    print(f"    Image  (B,W,C,H,W): Mean={img_mean:.4f} | Max={img_max:.4f} {'[WARNING: DARK]' if img_max < 0.1 else ''}")
                    print(f"    Proprio(B,W,8)    : Mean={prop_mean:.4f}")
                    # Print Quat
                    pq = proprio[0, -1, 3:7].tolist()
                    print(f"    PropQuat (xyzw)   : [{pq[0]:.3f}, {pq[1]:.3f}, {pq[2]:.3f}, {pq[3]:.3f}]")
                    print(f"    Goal   (B,64)     : Mean={goal_mean:.4f} | Target={cube_pos[0].tolist()}")
                # -------------------------

                # Model Forward Pass
                t0 = time.time()
                with torch.inference_mode():
                    # Disable queue updates during eval to match training config
                    # Returns: pred_action, next_latent, requery_logit
                    actions, _, _ = model(images, proprio, goal_embedding=goal_emb, update_queue=False, use_memory=False)
                dt = (time.time() - t0) * 1000
                
                # --- PROCESS ACTIONS (Model -> Sim) ---
                # Model Output is 8D (Pos [3], Quat [4], Gripper [1])
                # Env Expects (for Absolute IK): 7D Pose (Pos+Quat) + 1D Gripper = 8D ??
                # Wait, DifferentialInverseKinematicsActionCfg with command_type="pose" expects: 
                # (x, y, z, qw, qx, qy, qz) [7]
                # OR (x, y, z, qx, qy, qz, qw) ? Isaac Lab usually uses (w, x, y, z).
                # AxisModel output? "Rot: [0.741, ...]"
                # If AxisModel matches standard PyTorch 3D or similar, check component order.
                # Assuming model matches env for now.
                # We simply concat them.
                
                # env.step expects a single tensor if not dict?
                # ActionManager processes terms. 
                # Term 0: arm_action (7D) -> Pos (3) + Rot (4)
                # Term 1: gripper_action (1D)
                # We need to construct the full action tensor.
                
                # actions is (B, 8)
                # Model Output: [x, y, z, qx, qy, qz, qw, g] (Scalar Last)
                # Env Expects:  [x, y, z, qw, qx, qy, qz]    (Scalar First) + Gripper
                
                # 1. Split
                act_pos = actions[:, :3]
                act_quat_xyzw = actions[:, 3:7]
                act_grip = actions[:, 7:]
                
                # 2. Permute Quat xyzw -> wxyz
                # w=3, x=0, y=1, z=2
                act_quat_wxyz = torch.cat([act_quat_xyzw[:, 3:4], act_quat_xyzw[:, :3]], dim=1)
                
                # 3. Handle Gripper (Model Logit -> Sigmoid -> Sim -1/1)
                # Model output is UNBOUNDED (Linear). We need Sigmoid first.
                act_grip_logit = act_grip
                act_grip_prob = torch.sigmoid(act_grip_logit)
                
                # Hard Threshold for Binary Sim Gripper
                # Prob > 0.5 -> Close (-1.0) SWAPPED
                # Prob <= 0.5 -> Open (1.0)  SWAPPED
                # Hypothesis: Sim Positive = Open, Negative = Close
                act_grip_cmd = torch.where(act_grip_prob > 0.5, 
                                          torch.tensor(-1.0, device=device), 
                                          torch.tensor(1.0, device=device))
                
                # 4. Reassemble
                env_actions = torch.cat([act_pos, act_quat_wxyz, act_grip_cmd], dim=1)
                
                # --- DEBUG: Print Action Stats ---
                log_freq = 10
                if i % log_freq == 0:
                    # actions: (B, 8) -> [x, y, z, qx, qy, qz, qw, g]
                    a = actions[0].tolist()
                    # Check if inputs are changing
                    p_diff = proprio.std().item()
                    # Raw proprio
                    raw_p = proprio[0, -1].tolist()
                    
                    # Distance to Target
                    dist = np.linalg.norm(current_pose_np[:3] - cube_pos_np)
                    
                    print(f"[{i}] {dt:.1f}ms | InDiff: {p_diff:.4f} | Dist: {dist:.4f}") 
                    print(f"      Proprio: Pos=[{raw_p[0]:.3f}, {raw_p[1]:.3f}, {raw_p[2]:.3f}]")
                    # Log Probability and Final Command
                    g_prob = act_grip_prob[0].item()
                    g_cmd = act_grip_cmd[0].item()
                    print(f"      Cmd:     Pos=[{a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f}] | GripProb={g_prob:.3f} -> {g_cmd}")
                # -------------------------
                
            obs, reward, terminated, truncated, info = env.step(env_actions)
            
            dt = (time.time() - t0) * 1000
            
            if terminated.any() or truncated.any():
                print(f"!!! RESET triggered at step {i} (Terminated: {terminated.any()}, Truncated: {truncated.any()}) !!!")
                obs, info = env.reset()

        print("Evaluation Complete.")
    except Exception as e:
        print(f"Runtime error during eval: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close App
        if 'env' in locals():
            env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()
