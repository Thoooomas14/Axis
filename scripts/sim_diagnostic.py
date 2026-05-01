"""
Diagnostic tool: Replay training data in Isaac Sim and compare with Model Predictions.

Purpose:
1. Verify coordinate frame alignment between dataset and simulation.
2. Debug simulation-to-real drift by comparing GT proprioception with SIM proprioception.
3. Perform "Teacher Forcing" evaluation (evaluating model on dataset images but sim proprio).

Usage:
    # Replay Ground Truth only:
    python scripts/sim_diagnostic.py --local_data_path path/to/data.h5 --mode gt

    # Comparison (Teacher Forcing):
    python scripts/sim_diagnostic.py --checkpoint path/to/model --mode both --proprio_source sim
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
import pypose as pp

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 0. Bootstrap Isaac Sim
try:
    import isaacsim  # type: ignore # noqa: F401
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

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
isaac_sim_path = os.path.join(project_root, "isaac_sim")
if os.path.exists(isaac_sim_path) and isaac_sim_path not in sys.path:
    sys.path.append(isaac_sim_path)

try:
    from omni.isaac.lab.app import AppLauncher  # type: ignore
except ImportError:
    try:
        from isaaclab.app import AppLauncher  # type: ignore
    except ImportError as e:
        print(f"CRITICAL ERROR: Failed to import AppLauncher. {e}")
        sys.exit(1)

# =========================================================================
# ARGPARSE + APP LAUNCH
# =========================================================================
parser = argparse.ArgumentParser(description="Unified Isaac Sim Diagnostic tool")

parser.add_argument(
    "--local_data_path",
    type=str,
    default=None,
    help="Path to preprocessed HDF5 file (overrides --dataset)",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to model checkpoint (for --mode predicted/both)",
)
parser.add_argument("--episode", type=int, default=0, help="Episode index to replay")
parser.add_argument(
    "--mode",
    type=str,
    default="gt",
    choices=["gt", "predicted", "both", "diagnostic"],
    help="gt=replay ground truth, predicted=replay model predictions, "
    "both=print comparison, diagnostic=print data without sim",
)
parser.add_argument(
    "--image_source",
    type=str,
    default="dataset",
    choices=["dataset", "sim", "black"],
    help="Source of images for predicted/both modes",
)
parser.add_argument(
    "--proprio_source",
    type=str,
    default="dataset",
    choices=["dataset", "sim"],
    help="Source of proprioception for predicted/both modes",
)
parser.add_argument(
    "--goal_source",
    type=str,
    default=None,
    choices=["dataset", "sim"],
    help="Source for goal vector start pose. Defaults to --proprio_source if not set.",
)
parser.add_argument("--steps", type=int, default=200, help="Max steps to replay")
parser.add_argument(
    "--speed",
    type=float,
    default=1.0,
    help="Replay speed multiplier (0.5 = half speed)",
)
parser.add_argument(
    "--start_frame", type=int, default=0, help="Starting frame in the episode"
)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Only launch sim if not in diagnostic mode
if args.mode != "diagnostic":
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

# =========================================================================
# POST-LAUNCH IMPORTS
# =========================================================================
import h5py  # noqa: E402
from src.inference import AxisInference  # noqa: E402
from imitation.data.goal_oracle import GoalOracle  # noqa: E402
from my_robot_ext.tasks.eval_env import AxisEvalEnv, AxisEvalEnvCfg  # noqa: E402
from my_robot_ext.wrappers.axis_wrapper import AxisObservationWrapper  # noqa: E402
from my_robot_ext.config.robots import FrankaCfg  # noqa: E402

# =========================================================================
# HELPERS
# =========================================================================


def pose_13d_to_sim_action(pose_13d):
    """Converts 13D [R(9), pos(3), grip(1)] to Sim [pos(3), quat_wxyz(4), grip(1)]."""
    pos_mm = pose_13d[9:12]
    rot9 = pose_13d[:9].reshape(3, 3)
    grip = pose_13d[12]

    pos_m = pos_mm / 1000.0

    # Orthonormalize and convert to quat
    U, _, Vt = np.linalg.svd(rot9)
    det = np.linalg.det(U @ Vt)
    correction = np.eye(3)
    correction[2, 2] = det
    rot9_clean = U @ correction @ Vt

    q_xyzw = pp.mat2SO3(torch.tensor(rot9_clean, dtype=torch.float32)).tensor().numpy()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    # Standard: 1.0 = Open, -1.0 = Closed
    grip_cmd = 1.0 if grip > 0.5 else -1.0
    return pos_m, q_wxyz, grip_cmd, q_xyzw


def load_episode_local(data_path, episode_idx):
    with h5py.File(data_path, "r") as f:
        key = sorted(f.keys())[episode_idx]
        ep = f[key]
        imgs = np.array(ep["images"])
        props = np.array(ep["proprio"]).astype(np.float32)
        props[:, 9:12] *= 1000.0  # Scale to mm
        return imgs, props


def load_episode_rtx(dataset_name, data_dir, episode_idx):
    from imitation.data.rtx_stream_loader import RTXStreamLoader

    loader = RTXStreamLoader(
        dataset_name=dataset_name, window_size=2, data_dir=data_dir, repeat=False
    )
    all_props, all_imgs = [], []
    curr_ep, prev_prop = 0, None

    for batch in loader:
        p_win, i_win = batch["proprio"].numpy(), batch["images"].numpy()
        if prev_prop is not None and np.linalg.norm(prev_prop - p_win[0]) > 1.0:
            if curr_ep == episode_idx:
                break
            curr_ep += 1
            all_props, all_imgs = [], []

        if curr_ep == episode_idx:
            if not all_props:
                for j in range(p_win.shape[0]):
                    all_props.append(p_win[j])
                    all_imgs.append(i_win[j])
            else:
                all_props.append(p_win[-1])
                all_imgs.append(i_win[-1])
        prev_prop = p_win[-1]
    return np.array(all_imgs), np.array(all_props)


# =========================================================================
# CORE SIMULATION LOOP
# =========================================================================


def run_unified_replay(imgs, props, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Setup Env
    env_cfg = AxisEvalEnvCfg()
    env_cfg.robot = FrankaCfg()
    env = AxisEvalEnv(cfg=env_cfg, render_mode="rgb_array")
    env = AxisObservationWrapper(env, device=device)
    obs, _ = env.reset()

    # 2. Setup Agent if needed
    agent = None
    oracle = None
    goal_vector = None
    if args.mode in ["predicted", "both"]:
        agent = AxisInference(
            checkpoint_path=args.checkpoint,
            device=device,
            config="AxisV2",
            ensemble_k=0.0,
        )
        oracle = GoalOracle(output_dim=38)

        # Build Goal
        target_pose_13d = props[-1]
        start_pose = props[args.start_frame]

        # Guess task type: 1=Place(Open), 2=Pick(Close)
        task_type = 1 if (start_pose[12] < 0.5 and target_pose_13d[12] > 0.5) else 2

        # Burn-in to reach starting pose
        print(
            f"\n--- Burn-in Phase (60 steps) --- Target: {start_pose[9:12].round(1)} mm"
        )
        pos_m, q_wxyz, g_cmd, _ = pose_13d_to_sim_action(start_pose)
        burn_in_action = (
            torch.tensor(np.concatenate([pos_m, q_wxyz, [g_cmd]]), device=device)
            .float()
            .unsqueeze(0)
        )
        for _ in range(60):
            obs, _, _, _, _ = env.step(burn_in_action)

        # Wipe History
        env.image_buffer[:] = obs["images"][:, -1].unsqueeze(1)
        env.proprio_buffer[:] = obs["proprio"][:, -1].unsqueeze(1)
        obs["images"] = env.image_buffer
        obs["proprio"] = env.proprio_buffer
        agent.reset()

        # Finalize Goal from Burn-in state or Dataset
        goal_ref = (
            obs["proprio"][0, -1].cpu().numpy()
            if args.goal_source == "sim"
            else start_pose
        )
        goal_vector = oracle.encode_goal(task_type, goal_ref, target_pose_13d).numpy()
        print(f"Goal initialized (Task Type: {task_type})")

    # 3. Main Loop
    end_frame = min(args.start_frame + args.steps, len(props))
    delay = (1.0 / 30.0) / args.speed

    for i in range(args.start_frame, end_frame):
        # Prepare Windowed Input (Dataset)
        w_idx = slice(max(0, i - 9), i + 1)
        dat_imgs = imgs[w_idx]
        dat_props = props[w_idx]
        if len(dat_props) < 10:
            pad = 10 - len(dat_props)
            dat_imgs = np.concatenate([np.repeat(dat_imgs[:1], pad, 0), dat_imgs])
            dat_props = np.concatenate([np.repeat(dat_props[:1], pad, 0), dat_props])

        # Inference
        if agent:
            in_imgs = (
                obs["images"][0].cpu().numpy()
                if args.image_source == "sim"
                else dat_imgs
            )
            in_props = (
                obs["proprio"][0].cpu().numpy()
                if args.proprio_source == "sim"
                else dat_props
            )

            # Switch reset if transition just happened
            if args.proprio_source == "sim" and i == args.start_frame + 21:
                print("--- Switching to Sim Proprio --- Resetting Agent Cache...")
                agent.reset()

            res = agent.predict(in_imgs, in_props, goal_vector)
            act_pos_m = res["position"] / 1000.0
            act_q_xyzw = res["quaternion"]
            act_q_wxyz = np.array(
                [act_q_xyzw[3], act_q_xyzw[0], act_q_xyzw[1], act_q_xyzw[2]]
            )
            act_g_cmd = 1.0 if res["gripper"] > 0.5 else -1.0

        # Determine step action
        gt_pos_m, gt_q_wxyz, gt_g_cmd, _ = pose_13d_to_sim_action(props[i])

        if args.mode == "predicted":
            step_action = np.concatenate([act_pos_m, act_q_wxyz, [act_g_cmd]])
        else:  # gt or both
            # If in 'both' mode, we follow GT to maintain path but compare predictions
            step_action = np.concatenate([gt_pos_m, gt_q_wxyz, [gt_g_cmd]])

        # Step
        obs, _, _, _, _ = env.step(
            torch.tensor(step_action, device=device).float().unsqueeze(0)
        )

        # Log
        if i % 10 == 0:
            sim_pos_mm = obs["proprio"][0, -1, 9:12].cpu().numpy()
            print(
                f"Step {i:3d} | GT: {props[i, 9:12].round(1)} | SIM: {sim_pos_mm.round(1)} | Error: {np.linalg.norm(props[i, 9:12] - sim_pos_mm):.1f}mm"
            )
            if agent:
                print(
                    f"         | PRED: {(res['position']).round(1)} | Grip: {res['gripper']:.2f}"
                )

        time.sleep(delay)
    env.close()


def main():
    if args.local_data_path:
        imgs, props = load_episode_local(args.local_data_path, args.episode)
    else:
        imgs, props = load_episode_rtx(args.dataset, args.data_dir, args.episode)

    if args.mode == "diagnostic":
        print(f"Proprio Shape: {props.shape}")
        print(f"First Frame Pos: {props[0, 9:12]}")
        return

    run_unified_replay(imgs, props, args)
    simulation_app.close()


if __name__ == "__main__":
    main()
