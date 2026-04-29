import os
import sys
import torch
import argparse
import numpy as np

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.axis import AxisModel
from imitation.utils.visualizer import Visualizer
from imitation.data.rtx_stream_loader import RTXStreamLoader
from imitation.data.local_loader import LocalDataLoader


class TemporalEnsembler:
    """
    Simulates temporal ensembling by aggregating overlapping action chunks.

    Maintains a buffer of recent action chunks. For the current timestep t,
    it gathers all predictions covering t made at previous steps (t, t-1, t-2...).
    Then computes an exponentially weighted average.
    """

    def __init__(self, horizon, k=0.01):
        self.horizon = horizon
        self.k = k
        # List of (start_step, chunk_tensor)
        self.active_chunks = []

    def reset(self):
        self.active_chunks = []

    def update(self, start_step, chunk):
        """
        Add a new chunk prediction starting at start_step.
        chunk: (ChunkSize, ActionDim) tensor
        """
        if not isinstance(chunk, torch.Tensor):
            chunk = torch.tensor(chunk, dtype=torch.float32)

        self.active_chunks.append((start_step, chunk))

    def get_action(self, current_step):
        """
        Get the aggregated action for the current_step.
        """
        actions = []
        weights = []

        # Filter relevant chunks
        new_active = []
        for start_t, chunk in self.active_chunks:
            # Check if this chunk covers current_step
            # Index in chunk = current_step - start_t
            idx = current_step - start_t

            if idx >= 0 and idx < self.horizon:
                # Valid overlap
                actions.append(chunk[idx])
                # Exponential weight based on "freshness"
                w = np.exp(-self.k * idx)
                weights.append(w)
                new_active.append((start_t, chunk))
            elif idx < 0:
                # Future chunk?
                new_active.append((start_t, chunk))
            else:
                # Expired chunk
                pass

        self.active_chunks = new_active

        if not actions:
            # Fallback if no coverage
            return torch.zeros(7)

        # Weighted Average
        actions_stack = torch.stack(actions, dim=0)  # (N, 7)
        weights_stack = torch.tensor(weights, device=actions_stack.device).unsqueeze(
            1
        )  # (N, 1)

        weighted_sum = (actions_stack * weights_stack).sum(dim=0)
        total_weight = weights_stack.sum()

        return weighted_sum / total_weight


def make_gif(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- V2 Configuration ---
    config = "AxisV2"

    # --- Load Model ---
    model = AxisModel(config).to(device)
    config = model.config

    if args.random_weights:
        print("WARNING: Using Random Weights.")
        model.eval()
    else:
        model.eval()
        checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            # Try finding runs
            if os.path.exists(args.checkpoint_dir):
                files = [
                    f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pt")
                ]
                if files:
                    files.sort(
                        key=lambda x: os.path.getmtime(
                            os.path.join(args.checkpoint_dir, x)
                        )
                    )
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
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model at step {checkpoint.get('step', 'unknown')}")

    # --- Load Data ---
    if args.local_data_path:
        print(f"Loading from LOCAL HDF5: {args.local_data_path}")
        # If random, we load ALL episodes (max_episodes=0) and shuffle.
        # Otherwise, we take the first N (usually 1).
        local_max_episodes = (
            0 if args.random else (args.max_episodes if args.max_episodes > 0 else 1)
        )
        loader = LocalDataLoader(
            data_path=args.local_data_path,
            window_size=10,
            loss_horizon=10,
            shuffle=args.random,
            repeat=False,
            max_episodes=local_max_episodes,
        )
    else:
        print(f"Initializing RTXStreamLoader for {args.dataset}...")
        loader = RTXStreamLoader(
            dataset_name=args.dataset,
            split="train",
            window_size=10,
            image_size=(224, 224),
            data_dir=args.data_dir,
            repeat=False,
            use_subprocess=False,
            shuffle_buffer_size=0,
            shuffle_files=args.random,
            max_episodes=args.max_episodes if args.max_episodes > 0 else 1,
        )

    iterator = iter(loader)

    # --- Visualization Setup ---
    viz = Visualizer(args.checkpoint_dir)

    # --- Temporal Ensembling ---
    ensembler = TemporalEnsembler(horizon=config["chunk_size"], k=args.ensemble_k)

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

    # Initialize KV-Cache and Subtask Collection
    cached_tokens = None
    subtask_goal_poses_collected = []
    prev_proprio = None

    with torch.no_grad():
        for i in range(args.length):
            try:
                batch = next(iterator)
            except StopIteration:
                print("Dataset exhausted.")
                break

            if i % 10 == 0:
                print(f"Step {i}/{args.length}")

            # Unpack Batch
            images = batch["images"].to(device)
            if images.dim() == 4:
                images = images.unsqueeze(0)

            proprio = batch["proprio"].to(device)
            if proprio.dim() == 2:
                proprio = proprio.unsqueeze(0)

            # Check for discontinuity (new episode)
            # The loader yields windows with step 1.
            # So window[t][-1] (last frame) should equal window[t+1][-2] (second to last frame).
            if i > 0 and prev_proprio is not None:
                prev_last = prev_proprio[0, -1]
                curr_second_last = proprio[0, -2]
                # Compare
                if torch.linalg.norm(prev_last - curr_second_last) > 1e-3:
                    print(
                        f"Episode discontinuity detected at step {i} (Diff: {torch.linalg.norm(prev_last - curr_second_last)}). Stopping collection."
                    )
                    break

            prev_proprio = proprio

            goal = batch["goal"].to(device)
            if goal.dim() == 1:
                goal = goal.unsqueeze(0)

            # Initialize Prediction state if needed
            if curr_pred_pose_13d is None:
                # Start from the LAST proprio of the FIRST window?
                # Actually, make_gif usually iterates sequential windows.
                # The "current" state is the last item in the window input.
                curr_pred_pose_13d = proprio[0, -1].cpu().numpy()  # (13,)

                # Also store initial poses
                pred_poses.append(curr_pred_pose_13d)

            # Collect GT Pose (current state at end of window)
            gt_pose_curr = proprio[0, -1].cpu().numpy()  # (13,)
            gt_poses.append(gt_pose_curr)

            # Determine Proprio Input
            if args.closed_loop:
                if current_proprio_window is None:
                    # Initialize with GT from first batch
                    current_proprio_window = proprio.clone()
                proprio_input = current_proprio_window
            else:
                proprio_input = proprio

            # Collect Subtask Goal from Input Vector (Indices 16-29)
            # goal is (B, 38)
            current_subtask_goal = goal[0, 16:29].cpu().numpy()  # (13,)
            subtask_goal_poses_collected.append(current_subtask_goal)

            # Forward Pass with Caching
            import time

            start_time = time.time()

            pred_chunk, req_logit, cached_tokens = model(
                images,
                proprio_input,
                goal,
                cached_tokens=cached_tokens,
                return_tokens=True,
            )

            # Update KV-Cache: Keep last (W-1) tokens to form window for NEXT step
            # cached_tokens is (B, W, 512). We want (B, W-1, 512) representing indices 1..W
            if cached_tokens.shape[1] > 1:
                cached_tokens = cached_tokens[:, 1:, :]
            else:
                cached_tokens = cached_tokens  # Should not happen if W=10

            end_time = time.time()
            inference_times.append(end_time - start_time)

            # Debug Requery (First 5 steps)
            if i % 10 == 0:
                print(f"DEBUG: Step {i} Raw Requery Logit from Model: {req_logit}")

            # Action: Ensembling
            current_chunk = pred_chunk[0].cpu()  # (Chunk, 7)
            ensembler.update(i, current_chunk)

            # Get aggregated action for NOW (step i)
            current_pred_action = ensembler.get_action(i).unsqueeze(0)  # (1, 7)

            # Use raw logit as requested by user
            current_req = req_logit.cpu()  # (1, 1)

            pred_actions.append(current_pred_action)
            requery_preds.append(current_req)

            # Integrate Prediction
            integration_start_state = curr_pred_pose_13d
            integrated = viz.integrate_twist(
                integration_start_state, current_pred_action
            )
            next_pred_pose = integrated[0]

            curr_pred_pose_13d = next_pred_pose
            pred_poses.append(curr_pred_pose_13d)

            # Update proprio window for Closed Loop
            if args.closed_loop:
                next_pose_tensor = (
                    torch.tensor(next_pred_pose, device=device, dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                current_proprio_window = torch.cat(
                    [current_proprio_window[:, 1:], next_pose_tensor], dim=1
                )

            # Collect Visualization Data (Image)
            last_img = images[0, -1].cpu()  # (3, H, W)
            images_collected.append(last_img)

            # GT Twist (Future)
            if "actions" in batch:
                gt = batch["actions"][0, 0].cpu()  # (7,)
                gt_twists.append(gt)
            else:
                gt_twists.append(torch.zeros(7))

    if not images_collected:
        print("No data collected.")
        return

    # Stack results
    pred_actions_stack = torch.cat(pred_actions, dim=0)
    requery_preds_stack = torch.cat(requery_preds, dim=0)
    images_stack = torch.stack(images_collected, dim=0)
    gt_stack = torch.stack(gt_twists, dim=0)

    # Image[t], Twist[t], Pose[t] (state AT t)
    gt_poses_stack = torch.tensor(np.array(gt_poses), dtype=torch.float32)
    pred_poses_stack = torch.tensor(np.array(pred_poses[:-1]), dtype=torch.float32)
    subtask_goal_poses_tensor = torch.tensor(
        np.stack(subtask_goal_poses_collected), dtype=torch.float32
    )

    print(f"Collected {len(images_collected)} frames. Poses: {gt_poses_stack.shape}")

    # Debug Requery Stats
    if len(requery_preds) > 0:
        req_tensor = torch.stack(requery_preds)
        print(
            f"DEBUG: Requery Stats - Min: {req_tensor.min().item():.4f}, Max: {req_tensor.max().item():.4f}, Mean: {req_tensor.mean().item():.4f}"
        )
        if (req_tensor.max() - req_tensor.min()) < 1e-6:
            print(
                "DEBUG: Requery signal is static (model is likely outputting constant 0 logit)."
            )

    # --- Generate GIF ---
    print("Generating GIF...")
    viz.create_gif(
        0,
        images_stack,
        gt_stack,
        pred_actions_stack,
        gt_poses=gt_poses_stack,
        pred_poses=pred_poses_stack,
        requery_preds=requery_preds_stack,
        inference_times=inference_times,
        subtask_goal_poses=subtask_goal_poses_tensor,
        save_prefix="episode_stream_viz",
        dpi=100,
    )
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="droid")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument(
        "--local_data_path",
        type=str,
        default=None,
        help="Path to local HDF5 file (overrides streaming)",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=1,
        help="Max episodes to use (default: 1 for overfit testing)",
    )
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--random_weights", action="store_true", help="Use random weights"
    )
    parser.add_argument(
        "--random", action="store_true", help="Select from random episodes"
    )
    parser.add_argument(
        "--closed_loop",
        action="store_true",
        help="Use closed-loop proprioception feedback",
    )
    parser.add_argument(
        "--length", type=int, default=100, help="Number of steps to visualize"
    )
    parser.add_argument(
        "--ensemble_k",
        type=float,
        default=0.01,
        help="Exponential weighting decay for ensembling",
    )

    args = parser.parse_args()
    make_gif(args)
