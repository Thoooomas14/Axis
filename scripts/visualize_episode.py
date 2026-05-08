import os
import sys
import torch
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from imitation.data.local_loader import LocalDataLoader
from imitation.utils.visualizer import Visualizer


def visualize_episode(args):
    print("Visualizing Training Episode...")

    # --- Load Dataset ---

    print(f"Loading Local Dataset from {args.local_path}")
    dataset = LocalDataLoader(
        data_path=args.local_path,
        window_size=args.window_size,
        loss_horizon=1,
        repeat=False,  # One epoch
    )

    iterator = iter(dataset)
    try:
        batch = next(iterator)
    except StopIteration:
        print("Dataset is empty.")
        return

    # Unpack
    # Batch is a DICT now: {'images', 'proprio', 'goal', 'actions', 'target_poses'}
    print("Keys in batch:", batch.keys())

    # Create Visualizer
    viz = Visualizer(save_dir=".")

    # Fake a prediction (just use GT + noise for demo)
    gt_action = batch["actions"]  # (B, H, 7)
    # We visualizing the LAST step of the window
    # gt_action is (B, H, 7), we want (B, 7) for single step
    tgt_act = gt_action[:, 0, :]  # First horizon step

    # Fake Pred
    pred_act = tgt_act + torch.randn_like(tgt_act) * 0.1

    print("Visualizing Batch Window...")
    viz.visualize_batch(
        step=0, batch=batch, pred_action=pred_act, save_prefix="viz_debug"
    )

    print("Saved visualization to ./visualizations/viz_debug_step_0.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fractal20220817_data")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument(
        "--local_path",
        type=str,
        default=None,
        help="Path to .h5 file. If provided, uses LocalDataLoader.",
    )
    parser.add_argument("--window_size", type=int, default=10)

    args = parser.parse_args()
    visualize_episode(args)
