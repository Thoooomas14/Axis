# scripts/eval_isaac.py
import argparse
import os
import sys
import time

from gymnasium.wrappers import RecordVideo
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 0. Bootstrap Isaac Sim (Required for 4.0+)
try:
    import isaacsim
except ImportError:
    pass

# 1. Path Setup for Isaac Lab
cwd = os.getcwd()
possible_extension_paths = [
    os.path.join(cwd, "source"),
    os.path.join(cwd, "source", "isaaclab"),
]
for p in possible_extension_paths:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

# Add local extension path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
isaac_sim_path = os.path.join(project_root, "isaac_sim")

if os.path.exists(isaac_sim_path) and isaac_sim_path not in sys.path:
    print(f"Adding to sys.path: {isaac_sim_path}")
    sys.path.append(isaac_sim_path)

try:
    from omni.isaac.lab.app import AppLauncher
except ImportError:
    try:
        from isaaclab.app import AppLauncher
    except ImportError as e:
        print(f"CRITICAL ERROR: Failed to import AppLauncher. {e}")
        sys.exit(1)

# =========================================================================
# SETUP ARGPARSE AND LAUNCH THE SIMULATION APP GLOBALLY FIRST
# =========================================================================
parser = argparse.ArgumentParser(description="Evaluate Axis V2 in Isaac Lab")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
parser.add_argument("--robot", type=str, default="franka", choices=["franka", "google"], help="Robot type")
parser.add_argument("--steps", type=int, default=1000, help="Max steps")
parser.add_argument("--video", action="store_true", help="Record video")
parser.add_argument("--model_refresh", type=float, default=30, help="Control frequency Hz")
parser.add_argument("--chunk_size", type=int, default=10, help="Action chunk size used in model")
parser.add_argument("--random_weights", action="store_true", help="Use random weights")
parser.add_argument("--ignore_requery", action="store_true", help="Ignore model requery requests")
parser.add_argument("--ensemble_k", type=float, default=0.01, help="Exponential weighting decay")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch App before any Torch imports
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# =========================================================================
# NOW IMPORT PYTORCH AND CUSTOM MODULES (CUDA context is now safe)
# =========================================================================
import torch
from src.inference import AxisInference
from src.utils.rotation_utils import quaternion_to_rotation_6d
from imitation.data.goal_oracle import GoalOracle

from my_robot_ext.tasks.eval_env import AxisEvalEnv, AxisEvalEnvCfg
from my_robot_ext.wrappers.axis_wrapper import AxisObservationWrapper
from my_robot_ext.config.robots import FrankaCfg, GoogleRobotCfg


# =========================================================================
# GLOBAL HELPER FUNCTIONS
# =========================================================================
def quaternion_to_rot6d_numpy(quat_xyzw):
    """Helper to convert numpy quat [x,y,z,w] to 6D."""
    # Torch is globally available here!
    q_tensor = torch.tensor(quat_xyzw, dtype=torch.float32)
    r6 = quaternion_to_rotation_6d(q_tensor).numpy()
    return r6


# =========================================================================
# MAIN LOOP
# =========================================================================
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Axis V2 Inference ---
    print(f"Loading AxisInference from {args.checkpoint}...")
    # TODO: Save/load config from checkpoint for robustness with custom configs
    config = 'AxisV2'

    try:
        agent = AxisInference(
            args.checkpoint,
            config,
            device=device,
            random_weights=args.random_weights,
            ensemble_k=args.ensemble_k,
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load agent: {e}")
        simulation_app.close()
        sys.exit(1)

    oracle = GoalOracle(output_dim=38)  # Updated to 38D

    # --- Setup Environment ---
    try:
        env_cfg = AxisEvalEnvCfg()
        if args.robot == "franka":
            env_cfg.robot = FrankaCfg()
        elif args.robot == "google":
            env_cfg.robot = GoogleRobotCfg()

        render_mode = "rgb_array" if args.video else None
        env = AxisEvalEnv(cfg=env_cfg, render_mode=render_mode)

        if args.video:
            video_folder = os.path.join(os.path.dirname(args.checkpoint), "videos")
            env = RecordVideo(
                env,
                video_folder=video_folder,
                step_trigger=lambda step: step == 0,
                video_length=args.steps,
                name_prefix="eval_v2",
            )

        # Wrapper now returns 13D proprio
        # Use model's window_size to match training config
        env = AxisObservationWrapper(env, window_size=agent.config.get("window_size", 10), device=device)
    except Exception as e:
        print(f"Error initializing environment: {e}")
        simulation_app.close()
        return

    # --- Evaluation Loop ---
    try:
        obs, info = env.reset()
        agent.reset()  # Reset inference state (cache/ensembler)

        B = env.num_envs

        # State
        initial_pose_13d = None
        target_pos_sim = None
        goal_vector = np.zeros((B, 38), dtype=np.float32)  # 38D
        requery_flag = True
        requery_prob = 0.0
        last_pred_time = 0
        delta_pred_time = (1 / args.model_refresh) * 1000
        model_run = False
        last_goal_time = -100
        print("Starting Control Loop...")

        for i in range(args.steps):
            # 1. Get Observations (Torch -> Numpy)
            # Images: (B, W, C, H, W)
            images_np = obs["images"].cpu().numpy()
            # Proprio: (B, W, 13) [rot9, pos3, grip1]
            proprio_13d_np = obs["proprio"].cpu().numpy()

            # Use last proprio in window as current state
            current_pose_13d = proprio_13d_np[0, -1]  # (13,)

            # 3. Goal Logic (Sim specific hacking)
            if "cube" in env.unwrapped.scene.keys():
                cube_pos = (
                    env.unwrapped.scene["cube"].data.root_pos_w[0, :3].cpu().numpy()
                )
            else:
                cube_pos = np.zeros(3)

            if initial_pose_13d is None or requery_flag:
                initial_pose_13d = current_pose_13d.copy()

                # Goal Target: Cube Pos (converted to Model Frame)
                target_pos_model = cube_pos.copy()

                # Goal Pose: Target Pos + Fixed Downward Orientation
                # Construct 13D Target Pose
                target_pose_13d = np.zeros(13, dtype=np.float32)

                # Rotation: Downward [0,1,0,0] xyzw -> 180 deg around Y
                # This aligns the gripper opposite to X?
                target_quat_xyzw = np.array([0.0, 1.0, 0.0, 0.0])
                target_rot = R.from_quat(target_quat_xyzw).as_matrix().flatten()

                target_pose_13d[:9] = target_rot
                target_pose_13d[9:12] = (
                    target_pos_model * 1000.0
                )  # Meters -> Millimeters
                target_pose_13d[12] = 1.0  # Close gripper

                # Encode Goal
                # Task 1: Pick
                # Oracle handles 13D logic internally now
                # Define Object Props for Red Cube
                # These must match the simulated object in AxisSceneCfg
                obj_props = {
                    "color": np.array([1.0, 0.0, 0.0], dtype=np.float32),  # Red
                    "shape": np.array([1.0, 0.0, 0.0], dtype=np.float32),  # Cube
                    "size": np.array([0.05, 0.05, 0.05], dtype=np.float32),  # 5cm
                }
                g_emb = oracle.encode_goal(
                    1, initial_pose_13d, target_pose_13d, object_props=obj_props
                )
                goal_vector[0] = g_emb.numpy()

                print(f"Goal Updated! Target: {target_pos_model.round(3)}")
                requery_flag = False
                last_goal_time = i

            # 4. Predict
            # Call agent with single sample (remove batch dim)
            current_time = int(time.time() * 1000)
            model_run = False

            if (
                current_time - last_pred_time
            ) >= delta_pred_time or last_pred_time == 0:
                print("Model Prediction Called!", flush=True)
                model_run = True
                last_pred_time = current_time
                batch_result = agent.predict(
                    images_np[0],  # (W, C, H, W)
                    proprio_13d_np[0],  # (W, 13)
                    goal_vector[0],  # (38,)
                )
                requery_flag_model = batch_result.get("requery", False)
                requery_prob = batch_result.get("requery_prob", 0.0)
                # print(f" Position: {batch_result['position']}")

            if i == 0:
                # Debug: Save what the model sees
                import cv2

                # Image is (C, H, W) float 0-1. Convert to (H, W, C) uint8 0-255
                debug_img = images_np[0, -1].transpose(1, 2, 0)
                debug_img = (debug_img * 255).astype(np.uint8)
                # RGB -> BGR for OpenCV
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite("debug_view.png", debug_img)
                print(f"Saved debug view to {os.path.abspath('debug_view.png')}")
                print(
                    f"Image Stats: Min={images_np.min()}, Max={images_np.max()}, Mean={images_np.mean()}, Shape={images_np.shape}"
                )

            if i % 10 == 0:
                print(f"Requery Flag: {requery_flag_model} (Prob: {requery_prob:.3f})")

            if args.ignore_requery:
                requery_flag = False
            else:
                # Debounce Goal Updates to prevent loops
                # Only update if probability is high (>0.8) OR if we haven't updated in 50 steps
                # And ensure we don't spam updates every frame
                time_since_last_goal = i - last_goal_time

                if requery_flag_model and time_since_last_goal > 150:
                    requery_flag = True
                else:
                    requery_flag = False

            # 5. Execute
            # Model Output: 'action_absolute' is what Inference class returns.
            # Inference now returns 13D absolute pose in mm.

            act_pos_mm = batch_result["position"]  # (3,) mm
            act_quat_xyzw = batch_result["quaternion"]  # (4,)
            act_grip = batch_result["gripper"]

            # Flip Y for Sim (Model -> Sim)
            # AND Convert mm -> meters
            act_pos_sim = act_pos_mm.copy() / 1000.0
            # act_pos_sim[1] *= -1.0 (Disabled)

            # Quat conversion xyzw -> wxyz for Isaac
            act_quat_wxyz = np.array(
                [act_quat_xyzw[3], act_quat_xyzw[0], act_quat_xyzw[1], act_quat_xyzw[2]]
            )

            # Gripper Cmd — DROID convention: model output > 0.5 means "open"
            # BinaryJointPositionAction: -1.0 = open, 1.0 = close
            act_grip_cmd = -1.0 if act_grip > 0.5 else 1.0

            # Compose
            env_action = np.concatenate([act_pos_sim, act_quat_wxyz, [act_grip_cmd]])
            env_action_t = torch.tensor(env_action, device=device).unsqueeze(
                0
            )  # (1, 8)

            # Step
            obs, reward, terminated, truncated, info = env.step(env_action_t, model_run)

            if i % 10 == 0:
                print(f"Step {i} --------------------------------------------------")
                print(f"  Current Pos (mm): {current_pose_13d[9:12].round(1)}")
                r_euler = R.from_quat(act_quat_xyzw).as_euler("xyz", degrees=True)
                print(
                    f"  Action Twist (Lin/Ang): {batch_result['action_twist'][:3].round(3)} / {batch_result['action_twist'][3:6].round(3)}"
                )
                print(f"  Target  Pos (m) : {act_pos_sim.round(3)}")
                print(f"  Target  Rot (deg): {r_euler.round(1)} (XYZ)")
                print(
                    f"  Target  Grip    : {act_grip:.3f} ({'Open' if act_grip > 0.5 else 'Close'})"
                )
                print(f"  Requery Flag    : {requery_flag_model}")
                if terminated.any() or truncated.any():
                    print(f"!!! RESET TRIGGERED (Step {i}) !!!", flush=True)
                    print(f"  Terminated: {terminated}", flush=True)
                    print(f"  Truncated: {truncated}", flush=True)
                    print(f"  Info: {info}", flush=True)

                    obs, info = env.reset()
                    agent.reset()  # Reset inference state
                    initial_pose_13d = None
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
