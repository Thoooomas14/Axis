"""
Diagnostic tool: Replay GROUND TRUTH training data in Isaac Sim.

Purpose: Determine if the model's training data coordinate frame matches
Isaac Sim's coordinate frame. If the robot follows the GT trajectory
correctly, the coordinate frames match and the issue is model inference.
If not, there's a fundamental frame mismatch.

Supports both local HDF5 and RTX streaming data sources.

Usage:
    # Local HDF5:
    python scripts/replay_training_in_sim.py \
        --local_data_path path/to/data.h5 --episode 0 --mode gt

    # RTX Streaming:
    python scripts/replay_training_in_sim.py \
        --dataset droid --mode gt

    # Diagnostic (no sim launch, just print data):
    python scripts/replay_training_in_sim.py \
        --local_data_path path/to/data.h5 --mode diagnostic
"""

import argparse
import os
import sys
import time

from scipy.spatial.transform import Rotation as R
import numpy as np
import torch

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
# ARGPARSE + APP LAUNCH (before any Torch/CUDA)
# =========================================================================
parser = argparse.ArgumentParser(description="Replay training data in Isaac Sim")

# Data source — defaults to RTX streaming from DROID
parser.add_argument(
    "--local_data_path",
    type=str,
    default=None,
    help="Path to preprocessed HDF5 file (overrides --dataset)",
)
parser.add_argument(
    "--dataset", type=str, default="droid", help="RTX dataset name (default: 'droid')"
)

parser.add_argument(
    "--data_dir", type=str, default=None, help="Data directory for RTX datasets"
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
from scipy.spatial.transform import Rotation as R  # noqa: E402, F811

# Suppress TF logs for RTX loading
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

if args.mode != "diagnostic":
    from my_robot_ext.tasks.eval_env import AxisEvalEnv, AxisEvalEnvCfg  # type: ignore
    from my_robot_ext.config.robots import FrankaCfg  # type: ignore


def load_episode_local(data_path, episode_idx):
    """Load a single episode from HDF5, returning poses in mm (same as training)."""
    with h5py.File(data_path, "r") as f:
        episode_keys = sorted(f.keys())
        if episode_idx >= len(episode_keys):
            print(f"Episode {episode_idx} not found. Max: {len(episode_keys) - 1}")
            sys.exit(1)

        key = episode_keys[episode_idx]
        ep = f[key]
        imgs = np.array(ep["images"])  # (T, C, H, W) uint8
        props = np.array(ep["proprio"])  # (T, 13) float16/32

        # Scale positions to mm (same as LocalDataLoader)
        props = props.astype(np.float32)
        props[:, 9:12] *= 1000.0

        print(f"Loaded LOCAL episode '{key}': {len(props)} frames")
        _print_episode_summary(props)
        return imgs, props


def load_episode_rtx(dataset_name, data_dir, episode_idx, max_frames=400):
    """
    Load a single episode from RTX streaming dataset.

    Extracts a full episode by using RTXStreamLoader with window_size=2
    and stitching sequential windows by their last frame.
    """
    from imitation.data.rtx_stream_loader import RTXStreamLoader

    print(f"Loading RTX episode {episode_idx} from '{dataset_name}'...")

    # Use window_size=2, loss_horizon=1 to get minimal windows
    # that let us reconstruct the full episode trajectory
    loader = RTXStreamLoader(
        dataset_name=dataset_name,
        split="train",
        window_size=2,
        loss_horizon=1,
        image_size=(224, 224),
        data_dir=data_dir,
        repeat=False,
        use_subprocess=False,
        shuffle_buffer_size=0,
        shuffle_files=False,
        max_episodes=episode_idx + 1,  # Load up to the target episode
    )

    # Iterate and collect frames from the target episode
    all_props = []
    all_imgs = []
    prev_prop = None
    current_episode = 0

    for batch in loader:
        prop_window = batch["proprio"].numpy()  # (W, 13)
        img_window = batch["images"].numpy()  # (W, C, H, W)

        # Detect episode boundary (discontinuity)
        if prev_prop is not None:
            diff = np.linalg.norm(prev_prop - prop_window[0])
            if diff > 1.0:  # New episode
                if current_episode == episode_idx:
                    break  # We've finished our target episode
                current_episode += 1
                all_props = []
                all_imgs = []

        if current_episode == episode_idx:
            if len(all_props) == 0:
                # First window: add all frames
                for j in range(prop_window.shape[0]):
                    all_props.append(prop_window[j])
                    all_imgs.append(img_window[j])
            else:
                # Subsequent windows: add only the last frame (new one)
                all_props.append(prop_window[-1])
                all_imgs.append(img_window[-1])

        prev_prop = prop_window[-1]

        if len(all_props) >= max_frames:
            break

    if len(all_props) == 0:
        print(f"Episode {episode_idx} not found in dataset.")
        sys.exit(1)

    props = np.array(all_props, dtype=np.float32)  # Already in mm from loader
    imgs = np.array(all_imgs, dtype=np.float32)

    print(f"Loaded RTX episode {episode_idx}: {len(props)} frames")
    _print_episode_summary(props)
    return imgs, props


def _print_episode_summary(props):
    """Print summary statistics of an episode's proprioception."""
    print(
        f"  Position range (mm): X[{props[:, 9].min():.1f}, {props[:, 9].max():.1f}] "
        f"Y[{props[:, 10].min():.1f}, {props[:, 10].max():.1f}] "
        f"Z[{props[:, 11].min():.1f}, {props[:, 11].max():.1f}]"
    )
    print(f"  Gripper range: [{props[:, 12].min():.3f}, {props[:, 12].max():.3f}]")

    # Verify rotation matrices
    R_sample = props[0, :9].reshape(3, 3)
    det = np.linalg.det(R_sample)
    print(f"  R[0] determinant: {det:.6f} (should be ~1.0)")
    print(f"  R[0]:\n{R_sample.round(3)}")


def pose_13d_to_sim_action(pose_13d):
    """
    Convert a 13D training pose [rot9, pos_mm, grip] to Isaac Sim action [pos_m, quat_wxyz, grip_cmd].

    This is the CRITICAL conversion — if this is wrong, the replay won't match.
    """
    rot9 = pose_13d[:9].reshape(3, 3)
    pos_mm = pose_13d[9:12]
    grip = pose_13d[12]

    # Position: mm -> meters
    pos_m = pos_mm / 1000.0

    # Rotation matrix -> quaternion (scipy returns xyzw)
    # Orthonormalize first to handle float16 artifacts
    U, S, Vt = np.linalg.svd(rot9)
    det = np.linalg.det(U @ Vt)
    correction = np.eye(3)
    correction[2, 2] = det
    rot9_clean = U @ correction @ Vt

    quat_xyzw = R.from_matrix(rot9_clean).as_quat()  # [x, y, z, w]

    # Convert to wxyz for Isaac Sim
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    # Gripper: DROID convention (1=open, 0=closed) -> Isaac (1=open, -1=close)
    grip_cmd = 1.0 if grip > 0.5 else -1.0

    return pos_m, quat_wxyz, grip_cmd, quat_xyzw


def print_diagnostics(props):
    """Print detailed diagnostics about the training data without launching sim."""
    print("\n" + "=" * 70)
    print("TRAINING DATA DIAGNOSTIC")
    print("=" * 70)

    # Frame 0 details
    print("\n--- Frame 0 ---")
    p = props[0]
    R0 = p[:9].reshape(3, 3)
    pos0 = p[9:12]
    grip0 = p[12]
    euler0 = R.from_matrix(R0).as_euler("xyz", degrees=True)
    quat0 = R.from_matrix(R0).as_quat()

    print(f"  Rotation Matrix:\n{R0.round(4)}")
    print(f"  Euler (deg, XYZ): {euler0.round(1)}")
    print(f"  Quaternion (xyzw): {quat0.round(4)}")
    print(f"  Position (mm): {pos0.round(1)}")
    print(f"  Position (m):  {(pos0 / 1000).round(4)}")
    print(f"  Gripper: {grip0:.4f} ({'Open' if grip0 > 0.5 else 'Closed'})")

    # Position trajectory summary
    print(f"\n--- Trajectory Summary ({len(props)} frames) ---")
    pos = props[:, 9:12]
    print(f"  Start pos (m): {(pos[0] / 1000).round(4)}")
    print(f"  End pos (m):   {(pos[-1] / 1000).round(4)}")
    print(f"  Total displacement (mm): {np.linalg.norm(pos[-1] - pos[0]):.1f}")

    # Movement per frame
    deltas = np.diff(pos, axis=0)
    delta_norms = np.linalg.norm(deltas, axis=1)
    print(f"  Mean step size (mm): {delta_norms.mean():.3f}")
    print(f"  Max step size (mm):  {delta_norms.max():.3f}")

    # Gripper transitions
    grip = props[:, 12]
    transitions = np.where(np.abs(np.diff(grip)) > 0.3)[0]
    print("\n--- Gripper Transitions ---")
    if len(transitions) > 0:
        for t in transitions[:10]:  # Show first 10
            print(
                f"  Frame {t}: {grip[t]:.3f} -> {grip[t + 1]:.3f} "
                f"({'Open->Close' if grip[t] > grip[t + 1] else 'Close->Open'})"
            )
    else:
        print("  No significant gripper transitions found")

    # Compare with Isaac Sim's expected workspace
    print("\n--- Isaac Sim Compatibility ---")
    print("  Franka default EE position (m): ~(0.3-0.6, -0.3-0.3, 0.1-0.6)")
    print(
        f"  Training data EE position (m):  ({(pos[:, 0].min() / 1000):.3f}-{(pos[:, 0].max() / 1000):.3f}, "
        f"{(pos[:, 1].min() / 1000):.3f}-{(pos[:, 1].max() / 1000):.3f}, "
        f"{(pos[:, 2].min() / 1000):.3f}-{(pos[:, 2].max() / 1000):.3f})"
    )

    pos_m = pos / 1000.0
    in_workspace = (
        (pos_m[:, 0] > 0.1)
        & (pos_m[:, 0] < 0.8)
        & (pos_m[:, 1] > -0.5)
        & (pos_m[:, 1] < 0.5)
        & (pos_m[:, 2] > 0.0)
        & (pos_m[:, 2] < 0.8)
    )
    print(
        f"  Frames within Franka workspace: {in_workspace.sum()}/{len(pos_m)} "
        f"({100 * in_workspace.mean():.0f}%)"
    )

    # Show what Isaac Sim would receive for frame 0
    pos_sim, quat_wxyz, grip_cmd, quat_xyzw = pose_13d_to_sim_action(props[0])
    print("\n--- Frame 0 → Isaac Sim Action ---")
    print(f"  pos (m):       {pos_sim.round(4)}")
    print(f"  quat (wxyz):   {quat_wxyz.round(4)}")
    print(f"  quat (xyzw):   {quat_xyzw.round(4)}")
    print(f"  grip cmd:      {grip_cmd}")
    euler_sim = R.from_quat(quat_xyzw).as_euler("xyz", degrees=True)
    print(f"  euler (deg):   {euler_sim.round(1)}")


def replay_in_sim(props, start_frame, max_steps, speed):
    """Replay ground truth training trajectory in Isaac Sim."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup environment (NO observation wrapper — we're commanding directly)
    env_cfg = AxisEvalEnvCfg()
    env_cfg.robot = FrankaCfg()
    env = AxisEvalEnv(cfg=env_cfg, render_mode=None)

    obs, info = env.reset()
    print(f"\n{'=' * 70}")
    print("REPLAYING GROUND TRUTH TRAJECTORY IN ISAAC SIM")
    print(f"{'=' * 70}")

    # Read initial sim state for comparison
    policy_obs = obs.get("policy", {})
    ee_pose = policy_obs.get("ee_pose", None)
    if ee_pose is not None:
        sim_pos = ee_pose[0, :3].cpu().numpy()
        sim_quat_wxyz = ee_pose[0, 3:7].cpu().numpy()
        sim_euler = R.from_quat(
            [sim_quat_wxyz[1], sim_quat_wxyz[2], sim_quat_wxyz[3], sim_quat_wxyz[0]]
        ).as_euler("xyz", degrees=True)
        print(f"  Sim initial EE pos (m):   {sim_pos.round(4)}")
        print(f"  Sim initial EE euler:     {sim_euler.round(1)}")

    # Training data frame 0
    pos_train, quat_wxyz_train, _, quat_xyzw_train = pose_13d_to_sim_action(
        props[start_frame]
    )
    euler_train = R.from_quat(quat_xyzw_train).as_euler("xyz", degrees=True)
    print(f"  Training frame {start_frame} pos (m): {pos_train.round(4)}")
    print(f"  Training frame {start_frame} euler:   {euler_train.round(1)}")

    if ee_pose is not None:
        pos_diff = np.linalg.norm(sim_pos - pos_train) * 1000
        print(
            f"\n  ⚠ POSITION MISMATCH: {pos_diff:.1f} mm between sim and training data"
        )
        if pos_diff > 100:
            print(
                "  🔴 LARGE MISMATCH! This strongly suggests a coordinate frame issue."
            )
        elif pos_diff > 20:
            print("  🟡 Moderate mismatch — could be the body_offset (107mm)")
        else:
            print("  ✅ Positions are close")

    # Replay loop
    end_frame = min(start_frame + max_steps, len(props))
    delay = (1.0 / 30.0) / speed  # Assuming 30Hz training frequency

    for i in range(start_frame, end_frame):
        pose = props[i]
        pos_m, quat_wxyz, grip_cmd, quat_xyzw = pose_13d_to_sim_action(pose)

        # Compose action: [pos(3), quat_wxyz(4), grip(1)]
        env_action = np.concatenate([pos_m, quat_wxyz, [grip_cmd]])
        env_action_t = torch.tensor(
            env_action, device=device, dtype=torch.float32
        ).unsqueeze(0)

        obs, rew, terminated, truncated, info = env.step(env_action_t)

        # Read back sim state for comparison
        policy_obs = obs.get("policy", {})
        ee_pose = policy_obs.get("ee_pose", None)

        if i % 10 == 0:
            frame_idx = i - start_frame
            grip_str = "Open" if pose[12] > 0.5 else "Closed"
            print(f"\nStep {frame_idx:3d} (Frame {i})")
            print(
                f"  CMD  pos (m): {pos_m.round(4)}  euler: {R.from_quat(quat_xyzw).as_euler('xyz', degrees=True).round(1)}  grip: {grip_str}"
            )

            if ee_pose is not None:
                sim_pos = ee_pose[0, :3].cpu().numpy()
                sim_quat_wxyz = ee_pose[0, 3:7].cpu().numpy()
                sim_euler = R.from_quat(
                    [
                        sim_quat_wxyz[1],
                        sim_quat_wxyz[2],
                        sim_quat_wxyz[3],
                        sim_quat_wxyz[0],
                    ]
                ).as_euler("xyz", degrees=True)
                err = np.linalg.norm(sim_pos - pos_m) * 1000
                print(
                    f"  SIM  pos (m): {sim_pos.round(4)}  euler: {sim_euler.round(1)}  err: {err:.1f}mm"
                )

        if terminated.any() or truncated.any():
            print(f"\n!!! SIM RESET at step {i - start_frame}")
            break

        time.sleep(delay)

    print(f"\nReplay complete. {end_frame - start_frame} frames replayed.")
    env.close()


def replay_predicted_in_sim(
    imgs,
    props,
    checkpoint_dir,
    start_frame,
    max_steps,
    speed,
    image_source="dataset",
    proprio_source="dataset",
    goal_source=None,
):
    """Replay open-loop model predictions in Isaac Sim using specified inputs."""
    import time
    from src.inference import AxisInference
    from imitation.data.goal_oracle import GoalOracle
    from isaac_sim.my_robot_ext.wrappers.axis_wrapper import AxisObservationWrapper

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup environment
    env_cfg = AxisEvalEnvCfg()
    env_cfg.robot = FrankaCfg()
    env = AxisEvalEnv(cfg=env_cfg, render_mode="rgb_array")
    env = AxisObservationWrapper(env, device=device)

    obs, info = env.reset()
    print(f"\n{'=' * 70}")
    print(
        f"REPLAYING PREDICTIONS (Image: {image_source.upper()}, Proprio: {proprio_source.upper()})"
    )
    print(f"{'=' * 70}")

    # Initialize Agent
    print(f"Loading checkpoint from: {checkpoint_dir} ...")
    agent = AxisInference(
        checkpoint_path=checkpoint_dir, device=device, config="AxisV2"
    )

    # Encode Goal
    oracle = GoalOracle(output_dim=38)

    # Target pose is ALWAYS from the dataset
    target_pose_13d = props[-1]

    # --- BURN-IN PHASE ---
    print("\n" + "=" * 50)
    print("Starting Burn-in Phase to reach in-distribution pose...")
    print("=" * 50)

    # Use the helper function to get sim commands for the first frame
    burn_in_pos, burn_in_quat, burn_in_grip, _ = pose_13d_to_sim_action(
        props[start_frame]
    )
    burn_in_action = np.concatenate([burn_in_pos, burn_in_quat, [burn_in_grip]])
    burn_in_action_t = torch.tensor(
        burn_in_action, device=device, dtype=torch.float32
    ).unsqueeze(0)

    for b in range(60):
        obs, rew, terminated, truncated, info = env.step(burn_in_action_t)
        time.sleep(1.0 / 60.0)

    print("Burn-in complete. Wiping history and initializing Agent...")

    # Wipe the sliding windows to erase the burn-in movement
    last_image = obs["images"][:, -1]  # (1, C, H, W)
    last_proprio = obs["proprio"][:, -1]  # (1, 13)
    env.image_window = last_image.unsqueeze(1).repeat(1, agent.window_size, 1, 1, 1)
    env.proprio_window = last_proprio.unsqueeze(1).repeat(1, agent.window_size, 1)

    obs["images"] = env.image_window
    obs["proprio"] = env.proprio_window
    # ---------------------

    # Goal source defaults to proprio_source if not explicitly set
    effective_goal_source = goal_source if goal_source is not None else proprio_source

    # Goal vector start pose depends on goal_source
    if effective_goal_source == "sim":
        goal_initial_pose = obs["proprio"][0, -1].cpu().numpy()
    else:
        goal_initial_pose = props[start_frame]

    # Guess task type from gripper transition
    start_grip = props[start_frame][12]
    end_grip = target_pose_13d[12]
    task_type = 2  # Default to Pick (Close)
    if start_grip < 0.5 and end_grip > 0.5:
        task_type = 1  # Open -> Place
    elif start_grip > 0.5 and end_grip < 0.5:
        task_type = 2  # Close -> Pick

    g_emb = oracle.encode_goal(
        task_type, goal_initial_pose, target_pose_13d, object_props=None
    )
    goal_vector = g_emb.numpy()
    print(f"  Goal built from: {effective_goal_source.upper()} proprio")

    # --- GOAL VECTOR COMPARISON DIAGNOSTIC ---
    # Always compute both goal vectors so we can see the difference
    dat_initial = props[start_frame]
    sim_initial = obs["proprio"][0, -1].cpu().numpy()

    g_dat = oracle.encode_goal(
        task_type, dat_initial, target_pose_13d, object_props=None
    ).numpy()
    g_sim = oracle.encode_goal(
        task_type, sim_initial, target_pose_13d, object_props=None
    ).numpy()

    print(f"\n{'=' * 70}")
    print("GOAL VECTOR COMPARISON (Step 0)")
    print(f"{'=' * 70}")
    print(
        f"  Using:          {'SIM' if proprio_source == 'sim' else 'DATASET'} proprio for goal"
    )
    print(
        f"  Task type:      {task_type} ({'Move' if task_type == 0 else 'Pick' if task_type == 1 else 'Place'})"
    )
    print("")
    print("  --- Start Pose (goal[3:16]) ---")
    print(f"  DAT start rot:  {g_dat[3:12].round(3)}")
    print(f"  SIM start rot:  {g_sim[3:12].round(3)}")
    print(f"  DAT start pos:  {g_dat[12:15].round(1)} mm")
    print(f"  SIM start pos:  {g_sim[12:15].round(1)} mm")
    print(f"  DAT start grip: {g_dat[15]:.3f}")
    print(f"  SIM start grip: {g_sim[15]:.3f}")
    print("")
    print("  --- Target Pose (goal[16:29]) --- (should be identical)")
    print(f"  DAT target pos: {g_dat[25:28].round(1)} mm")
    print(f"  SIM target pos: {g_sim[25:28].round(1)} mm")
    print(f"  DAT target grip:{g_dat[28]:.3f}")
    print(f"  SIM target grip:{g_sim[28]:.3f}")
    print("")
    print("  --- Full Diff (|DAT - SIM|) ---")
    diff = np.abs(g_dat - g_sim)
    nonzero = np.where(diff > 0.001)[0]
    for idx in nonzero:
        print(
            f"    goal[{idx:2d}]: DAT={g_dat[idx]:+.4f}  SIM={g_sim[idx]:+.4f}  Δ={diff[idx]:.4f}"
        )
    if len(nonzero) == 0:
        print("    (identical)")
    print(f"{'=' * 70}\n")

    end_frame = min(start_frame + max_steps, len(props))
    delay = (1.0 / 30.0) / speed
    window_size = 10

    for i in range(start_frame, end_frame):
        # Create Sliding Window for dataset inputs
        if i < window_size:
            w_start = 0
            w_end = i + 1
        else:
            w_start = i - window_size + 1
            w_end = i + 1

        current_imgs = imgs[w_start:w_end]
        current_props = props[w_start:w_end]

        # Pad dataset inputs if needed
        if len(current_imgs) < window_size:
            pad_len = window_size - len(current_imgs)
            current_imgs = np.concatenate(
                [np.repeat(current_imgs[:1], pad_len, axis=0), current_imgs]
            )
            current_props = np.concatenate(
                [np.repeat(current_props[:1], pad_len, axis=0), current_props]
            )

        # Select Image Source
        if image_source == "sim":
            final_imgs = obs["images"][0].cpu().numpy()
        elif image_source == "black":
            final_imgs = np.zeros_like(current_imgs)
        else:
            final_imgs = current_imgs

        # Select Proprio Source
        if proprio_source == "sim":
            final_props = obs["proprio"][0].cpu().numpy()
        else:
            final_props = current_props

        # Predict next action
        batch_result = agent.predict(
            final_imgs,
            final_props,
            goal_vector,
        )

        act_pos_mm = batch_result["position"]
        act_quat_xyzw = batch_result["quaternion"]
        act_grip = batch_result["gripper"]

        act_pos_sim = act_pos_mm.copy() / 1000.0
        act_quat_wxyz = np.array(
            [act_quat_xyzw[3], act_quat_xyzw[0], act_quat_xyzw[1], act_quat_xyzw[2]]
        )
        grip_cmd = -1.0 if act_grip > 0.5 else 1.0

        env_action = np.concatenate([act_pos_sim, act_quat_wxyz, [grip_cmd]])
        env_action_t = torch.tensor(
            env_action, device=device, dtype=torch.float32
        ).unsqueeze(0)

        # Step Simulation
        obs, rew, terminated, truncated, info = env.step(env_action_t)

        # Get GT action for comparison
        gt_pos_m, gt_quat_wxyz, gt_grip_cmd, _ = pose_13d_to_sim_action(props[i])

        if i % 10 == 0:
            frame_idx = i - start_frame
            print(f"\nStep {frame_idx:3d} (Frame {i})")
            print(f"  GT   pos (m): {gt_pos_m.round(4)}")
            print(f"  PRED pos (m): {act_pos_sim.round(4)}")
            err_pos = np.linalg.norm(gt_pos_m - act_pos_sim) * 1000
            print(f"  Pos Error:    {err_pos:.1f} mm")

        if terminated.any() or truncated.any():
            print(f"\n!!! SIM RESET at step {i - start_frame}")
            break

        time.sleep(delay)

    print(
        f"\nReplay complete. {end_frame - start_frame} frames predicted and replayed."
    )
    env.close()


def replay_both_in_sim(
    imgs,
    props,
    checkpoint_dir,
    start_frame,
    max_steps,
    speed,
    image_source="dataset",
    proprio_source="dataset",
    goal_source=None,
):
    """Replay ground truth in Sim, but run model inference at each step to compute 1-step prediction error (Teacher Forcing)."""
    import time
    from src.inference import AxisInference
    from imitation.data.goal_oracle import GoalOracle
    from isaac_sim.my_robot_ext.wrappers.axis_wrapper import AxisObservationWrapper

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup environment
    env_cfg = AxisEvalEnvCfg()
    env_cfg.robot = FrankaCfg()
    env = AxisEvalEnv(cfg=env_cfg, render_mode="rgb_array")
    env = AxisObservationWrapper(env, device=device)

    obs, info = env.reset()
    print(f"\n{'=' * 70}")
    print(
        f"REPLAYING BOTH (TEACHER FORCING) (Image: {image_source.upper()}, Proprio: {proprio_source.upper()})"
    )
    print(f"{'=' * 70}")

    # Initialize Agent
    print(f"Loading checkpoint from: {checkpoint_dir} ...")
    agent = AxisInference(
        checkpoint_path=checkpoint_dir, device=device, config="AxisV2"
    )

    # Encode Goal
    oracle = GoalOracle(output_dim=38)

    # Target pose is ALWAYS from the dataset
    target_pose_13d = props[-1]

    # --- BURN-IN PHASE ---
    print("\n" + "=" * 50)
    print("Starting Burn-in Phase to reach in-distribution pose...")
    print("=" * 50)

    burn_in_pos, burn_in_quat, burn_in_grip, _ = pose_13d_to_sim_action(
        props[start_frame]
    )
    burn_in_action = np.concatenate([burn_in_pos, burn_in_quat, [burn_in_grip]])
    burn_in_action_t = torch.tensor(
        burn_in_action, device=device, dtype=torch.float32
    ).unsqueeze(0)

    for b in range(60):
        obs, rew, terminated, truncated, info = env.step(burn_in_action_t)
        time.sleep(1.0 / 60.0)

    print("Burn-in complete. Wiping history and initializing Agent...")

    # Wipe the sliding windows to erase the burn-in movement
    last_image = obs["images"][:, -1]  # (1, C, H, W)
    last_proprio = obs["proprio"][:, -1]  # (1, 13)
    env.image_window = last_image.unsqueeze(1).repeat(1, agent.window_size, 1, 1, 1)
    env.proprio_window = last_proprio.unsqueeze(1).repeat(1, agent.window_size, 1)

    obs["images"] = env.image_window
    obs["proprio"] = env.proprio_window
    # ---------------------

    # Goal source defaults to proprio_source if not explicitly set
    effective_goal_source = goal_source if goal_source is not None else proprio_source

    # Goal vector start pose depends on goal_source
    if effective_goal_source == "sim":
        goal_initial_pose = obs["proprio"][0, -1].cpu().numpy()
    else:
        goal_initial_pose = props[start_frame]

    # Guess task type from gripper transition
    start_grip = props[start_frame][12]
    end_grip = target_pose_13d[12]
    task_type = 2  # Default to Pick (Close)
    if start_grip < 0.5 and end_grip > 0.5:
        task_type = 1  # Open -> Place
    elif start_grip > 0.5 and end_grip < 0.5:
        task_type = 2  # Close -> Pick

    g_emb = oracle.encode_goal(
        task_type, goal_initial_pose, target_pose_13d, object_props=None
    )
    goal_vector = g_emb.numpy()
    print(f"  Goal built from: {effective_goal_source.upper()} proprio")

    end_frame = min(start_frame + max_steps, len(props))
    delay = (1.0 / 30.0) / speed
    window_size = 10

    for i in range(start_frame, end_frame):
        # Create Sliding Window for Model (up to current frame i)
        if i < window_size:
            w_start = 0
            w_end = i + 1
        else:
            w_start = i - window_size + 1
            w_end = i + 1

        current_imgs = imgs[w_start:w_end]
        current_props = props[w_start:w_end]

        if len(current_imgs) < window_size:
            pad_len = window_size - len(current_imgs)
            current_imgs = np.concatenate(
                [np.repeat(current_imgs[:1], pad_len, axis=0), current_imgs]
            )
            current_props = np.concatenate(
                [np.repeat(current_props[:1], pad_len, axis=0), current_props]
            )

        # Select Image Source
        if image_source == "sim":
            final_imgs = obs["images"][0].cpu().numpy()
        elif image_source == "black":
            final_imgs = np.zeros_like(current_imgs)
        else:
            final_imgs = current_imgs

        # Select Proprio Source
        if proprio_source == "sim":
            final_props = obs["proprio"][0].cpu().numpy()
        else:
            final_props = current_props

        # Predict next action
        batch_result = agent.predict(
            final_imgs,
            final_props,
            goal_vector,
        )

        act_pos_mm = batch_result["position"]
        act_pos_sim_pred = act_pos_mm.copy() / 1000.0

        # Command the robot using the Ground Truth to keep it on track
        gt_pos_m, gt_quat_wxyz, gt_grip_cmd, _ = pose_13d_to_sim_action(props[i])

        env_action = np.concatenate([gt_pos_m, gt_quat_wxyz, [gt_grip_cmd]])
        env_action_t = torch.tensor(
            env_action, device=device, dtype=torch.float32
        ).unsqueeze(0)

        # Step Simulation
        obs, rew, terminated, truncated, info = env.step(env_action_t)

        if i % 10 == 0:
            frame_idx = i - start_frame
            print(f"\nStep {frame_idx:3d} (Frame {i})")
            print(f"  GT   pos (m): {gt_pos_m.round(4)}")
            print(f"  PRED pos (m): {act_pos_sim_pred.round(4)}")
            err_pos = np.linalg.norm(gt_pos_m - act_pos_sim_pred) * 1000
            print(f"  Pos Error:    {err_pos:.1f} mm")

            # --- PROPRIO COMPARISON DIAGNOSTIC ---
            sim_prop = obs["proprio"][0, -1].cpu().numpy()
            dat_prop = current_props[-1]
            print("  --- Proprioception Check ---")
            print(f"  SIM Pos (mm): {sim_prop[9:12].round(1)}")
            print(f"  DAT Pos (mm): {dat_prop[9:12].round(1)}")
            print(f"  SIM Grip:     {sim_prop[12]:.3f}")
            print(f"  DAT Grip:     {dat_prop[12]:.3f}")
            print(f"  SIM Rot Det:  {np.linalg.det(sim_prop[:9].reshape(3, 3)):.3f}")
            print(f"  SIM Rot:\n{sim_prop[:9].reshape(3, 3).round(3)}")
            print(f"  DAT Rot:\n{dat_prop[:9].reshape(3, 3).round(3)}")

        if terminated.any() or truncated.any():
            print(f"\n!!! SIM RESET at step {i - start_frame}")
            break

        time.sleep(delay)

    print(f"\nReplay complete. {end_frame - start_frame} frames analyzed.")
    env.close()


# =========================================================================
# MAIN
# =========================================================================
def main():
    # Load episode from appropriate source
    if args.local_data_path:
        imgs, props = load_episode_local(args.local_data_path, args.episode)
    else:
        imgs, props = load_episode_rtx(args.dataset, args.data_dir, args.episode)

    if args.mode == "diagnostic":
        print_diagnostics(props)
        return

    if args.mode == "gt":
        replay_in_sim(props, args.start_frame, args.steps, args.speed)
    elif args.mode == "both":
        if args.checkpoint is None:
            import sys

            print("ERROR: --checkpoint required for --mode both")
            sys.exit(1)
        print_diagnostics(props)
        replay_both_in_sim(
            imgs,
            props,
            args.checkpoint,
            args.start_frame,
            args.steps,
            args.speed,
            args.image_source,
            args.proprio_source,
            args.goal_source,
        )
    elif args.mode == "predicted":
        if args.checkpoint is None:
            import sys

            print("ERROR: --checkpoint required for --mode predicted")
            sys.exit(1)
        replay_predicted_in_sim(
            imgs,
            props,
            args.checkpoint,
            args.start_frame,
            args.steps,
            args.speed,
            args.image_source,
            args.proprio_source,
            args.goal_source,
        )

    if args.mode != "diagnostic":
        simulation_app.close()


if __name__ == "__main__":
    main()
