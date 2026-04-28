import os
import sys
import torch
import numpy as np
import argparse

# Add project root to path
# Assuming script is in scripts/ and project root is one level up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.getcwd())  # Safe fallback for running from root

from src.models.axis import AxisModel
from imitation.data.rtx_stream_loader import RTXStreamLoader
from imitation.data.local_loader import LocalDataLoader
from imitation.utils.visualizer import Visualizer


def evaluate_sequence(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Configuration ---
    config = {
        "device": device,
        "goal_dim": 38,
        "cond_dim": 256,
        "vision_feature_dim": 256,
        "num_vision_tokens": 8,
        "window_size": 8,
        "latent_dim": 256,
        "queue_size": 10,
        "embed_dim": 256,
        "num_heads": 4,
        "num_layers": 4,
        "proprio_dim": 13,  # [R(9), p(3), g(1)]
        "action_dim": 7,  # [v(3), w(3), g(1)]
        "use_action_chunking": True,
        "chunk_size": 1,  # Default
    }

    # --- Load Model ---
    model = AxisModel(config).to(device)

    if args.random_weights:
        print("WARNING: Using Random Weights (Untrained Model) for testing.")
        model.eval()
    else:
        model.eval()
        if not os.path.exists(args.checkpoint_dir):
            print(f"Checkpoint directory {args.checkpoint_dir} not found.")
            return

        checkpoints = [f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pt")]
        if not checkpoints:
            print("No checkpoints found.")
            return

        checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        latest_checkpoint = checkpoints[-1]
        checkpoint_path = os.path.join(args.checkpoint_dir, latest_checkpoint)

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Filter out latent_queue state if shapes don't match
        state_dict = checkpoint["model_state_dict"]
        model_state = model.state_dict()

        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state:
                if v.shape != model_state[k].shape:
                    print(
                        f"Skipping {k} due to shape mismatch: {v.shape} vs {model_state[k].shape}"
                    )
                    continue
                filtered_state_dict[k] = v

        model.load_state_dict(filtered_state_dict, strict=False)

    # --- Load Data ---
    print("Loading data...")
    if args.local_path:
        print(f"Loading local data from {args.local_path}")
        dataset = LocalDataLoader(
            data_path=args.local_path, batch_size=1, window_size=8, repeat=False
        )
    else:
        print(f"Loading streaming data {args.dataset}")
        dataset = RTXStreamLoader(
            dataset_name=args.dataset,
            split="train",
            window_size=8,
            image_size=(224, 224),
            data_dir=args.data_dir,
            repeat=False,  # Don't repeat for eval
            use_subprocess=False,  # In-process is sufficient for 1 episode
            shuffle_buffer_size=0,  # No buffer = direct streaming
            shuffle_files=False,  # Sequential read
        )

    # Get one batch
    iterator = iter(dataset)
    try:
        batch = next(iterator)
    except StopIteration:
        print("Dataset empty.")
        return

    # Unpack batch (torch tensors)
    images = batch["images"].to(device)  # (W, 3, H, W) or (B, W, 3, H, W)
    proprio = batch["proprio"].to(device)  # (W, 13) or (B, W, 13)
    goal = batch["goal"].to(device)  # (38) or (B, 38)
    gt_action = batch["actions"].to(device)  # (H, 7) or (B, H, 7)

    # Add Batch Dimension if missing
    if images.dim() == 4:
        images = images.unsqueeze(0)
    if proprio.dim() == 2:
        proprio = proprio.unsqueeze(0)
    if goal.dim() == 1:
        goal = goal.unsqueeze(0)
    if gt_action.dim() == 2:
        gt_action = gt_action.unsqueeze(0)

    # Check target_poses
    if "target_poses" in batch:
        target_poses = batch["target_poses"].to(device)
        if target_poses.dim() == 2:
            target_poses = target_poses.unsqueeze(0)
        batch["target_poses"] = target_poses

    # Update batch dict so visualizer gets correct shapes
    batch["images"] = images
    batch["proprio"] = proprio
    batch["goal"] = goal
    batch["actions"] = gt_action

    B, W = images.shape[0], images.shape[1]
    print(f"Loaded batch: Img {images.shape}, Prop {proprio.shape}, Goal {goal.shape}")

    # --- Prediction ---
    # Model V2 (AxisModel) typically doesn't need explicit memory reset if passing full window
    # or it handles caching internally if we pass cache. Here we just pass the window.

    with torch.no_grad():
        # Forward pass
        # model expects (B, W, ...)
        # Current AxisModel forward: (images, proprio, goal_embedding, cached_tokens=None, return_tokens=False)
        # Goal is (B, 38).

        output = model(
            images,
            proprio,
            goal,  # (B, 38)
        )

        # Output is (pred_action, requery_logit)
        if isinstance(output, tuple):
            if len(output) == 2:
                pred_action_chunk, requery = output
            elif len(output) == 3:
                pred_action_chunk, requery, _ = output
            else:
                print(f"Unexpected tuple length from model: {len(output)}")
                pred_action_chunk = output[0]  # Try first
        elif isinstance(output, dict):
            pred_action_chunk = output.get("action_chunks", output.get("action"))
        else:
            pred_action_chunk = output

        # pred_action_chunk: (B, ChunkSize, 7)
        # We pass the FULL chunk for multi-step visualization
        pred_action = pred_action_chunk  # (B, ChunkSize, 7)

    # --- Debug Stats ---
    gt_act_np = gt_action.cpu().numpy()
    pred_act_np = pred_action.cpu().numpy()

    print("\n--- Action Statistics ---")
    # Just stats for first step to avoid noise
    print(
        f"GT Action (Step 0)  | Mean: {np.mean(np.abs(gt_act_np[:, 0])):6f} | Max: {np.max(np.abs(gt_act_np[:, 0])):6f}"
    )
    print(
        f"Pred Action (Step 0)| Mean: {np.mean(np.abs(pred_act_np[:, 0])):6f} | Max: {np.max(np.abs(pred_act_np[:, 0])):6f}"
    )
    print("-------------------------\n")

    # --- Visualization ---
    print("Generating visualization...")
    viz = Visualizer(save_dir=".")
    viz.visualize_batch(
        step=0, batch=batch, pred_action=pred_action, save_prefix="eval_test"
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fractal20220817_data")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--local_path",
        type=str,
        default=None,
        help="Path to local .h5 file. Uses LocalLoader if set.",
    )
    parser.add_argument(
        "--random_weights",
        action="store_true",
        help="Initialize model with random weights (no checkpoint)",
    )

    args = parser.parse_args()
    evaluate_sequence(args)
