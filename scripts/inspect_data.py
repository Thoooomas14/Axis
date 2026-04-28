"""
Data Inspection Script

Inspect preprocessed HDF5 data OR streaming RTX data to understand position value ranges.
This helps diagnose if units are in meters, centimeters, or something else.

Usage:
    # Inspect local HDF5:
    python scripts/inspect_data.py --data_path E:/data/droid.h5

    # Inspect streaming DROID data (run this on VM with GCS access):
    python scripts/inspect_data.py --streaming --dataset droid --num_windows 100
"""

import os
import sys
import argparse
import numpy as np

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def analyze_positions(
    all_positions, all_grippers, all_position_deltas, all_twists=None
):
    """Analyze and print position statistics."""

    print(f"\n{'=' * 60}")
    print("=== AGGREGATE STATISTICS ===")
    print(f"{'=' * 60}\n")

    print("Position ranges (raw values):")
    print(f"  X: [{all_positions[:, 0].min():.6f}, {all_positions[:, 0].max():.6f}]")
    print(f"  Y: [{all_positions[:, 1].min():.6f}, {all_positions[:, 1].max():.6f}]")
    print(f"  Z: [{all_positions[:, 2].min():.6f}, {all_positions[:, 2].max():.6f}]")

    # Total workspace span
    x_span = all_positions[:, 0].max() - all_positions[:, 0].min()
    y_span = all_positions[:, 1].max() - all_positions[:, 1].min()
    z_span = all_positions[:, 2].max() - all_positions[:, 2].min()

    print("\nWorkspace span:")
    print(f"  X span: {x_span:.6f}")
    print(f"  Y span: {y_span:.6f}")
    print(f"  Z span: {z_span:.6f}")

    # Per-step movement
    if len(all_position_deltas) > 0:
        delta_norms = np.linalg.norm(all_position_deltas, axis=1)
        print("\nPer-step movement (position delta norms):")
        print(f"  Min:    {delta_norms.min():.8f}")
        print(f"  Max:    {delta_norms.max():.8f}")
        print(f"  Mean:   {delta_norms.mean():.8f}")
        print(f"  Median: {np.median(delta_norms):.8f}")
        print(f"  Std:    {delta_norms.std():.8f}")

    # Twist analysis (if available)
    if all_twists is not None and len(all_twists) > 0:
        print("\n=== TWIST (ACTION) STATISTICS ===")
        # Twists are [ω(3), v(3), gripper_delta(1)]
        angular = all_twists[:, :3]  # Angular velocity
        linear = all_twists[:, 3:6]  # Linear velocity
        gripper_delta = all_twists[:, 6]  # Gripper delta

        angular_norms = np.linalg.norm(angular, axis=1)
        linear_norms = np.linalg.norm(linear, axis=1)

        print("\nAngular velocity (ω) norms:")
        print(f"  Min:  {angular_norms.min():.8f}")
        print(f"  Max:  {angular_norms.max():.8f}")
        print(f"  Mean: {angular_norms.mean():.8f}")

        print("\nLinear velocity (v) norms:")
        print(f"  Min:  {linear_norms.min():.8f}")
        print(f"  Max:  {linear_norms.max():.8f}")
        print(f"  Mean: {linear_norms.mean():.8f}")

        print("\nGripper delta:")
        print(f"  Min:  {gripper_delta.min():.6f}")
        print(f"  Max:  {gripper_delta.max():.6f}")
        print(f"  Mean: {gripper_delta.mean():.6f}")

    # Unit interpretation
    print(f"\n{'=' * 60}")
    print("=== UNIT INTERPRETATION ===")
    print(f"{'=' * 60}\n")

    mean_step = delta_norms.mean() if len(all_position_deltas) > 0 else 0
    workspace = max(x_span, y_span, z_span)

    print(f"Mean step movement: {mean_step:.6f}")
    print(f"Max workspace span: {workspace:.6f}")

    if workspace < 2.0 and mean_step < 0.1:
        print("\n✓ Data appears to be in METERS")
        print(f"  Workspace ~{workspace * 1000:.1f} mm (reasonable for robot arm)")
        print(f"  Step movement ~{mean_step * 1000:.2f} mm/step")
        print("\n  Recommendation: Scale by 1000x to convert to MILLIMETERS")
    elif workspace < 200 and mean_step < 10:
        print("\n✓ Data appears to be in CENTIMETERS")
        print(f"  Workspace ~{workspace * 10:.1f} mm")
        print(f"  Step movement ~{mean_step * 10:.2f} mm/step")
        print("\n  Recommendation: Consider scaling x10 to MILLIMETERS")
    elif workspace < 2000 and mean_step < 100:
        print("\n✓ Data appears to be in MILLIMETERS")
        print(f"  Workspace ~{workspace:.1f} mm")
        print("  Scale looks correct for training.")
    else:
        print("\n⚠ Data scale is UNUSUAL")
        print("  Workspace seems too large for a robot arm")
        print("  Possible issue with data preprocessing or units")

    # Gripper stats
    print(
        f"\nGripper state range: [{all_grippers.min():.4f}, {all_grippers.max():.4f}]"
    )
    if all_grippers.max() <= 1.1 and all_grippers.min() >= -0.1:
        print("  ✓ Gripper appears normalized [0, 1]")
    else:
        print("  ⚠ Gripper may need normalization")


def inspect_hdf5(data_path: str, num_episodes: int = 10):
    """Inspect HDF5 file to understand position scales."""
    import h5py

    print(f"\n{'=' * 60}")
    print(f"Inspecting HDF5: {data_path}")
    print(f"{'=' * 60}\n")

    with h5py.File(data_path, "r") as f:
        # Print metadata
        print("=== File Metadata ===")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")

        # Get episode keys
        episode_keys = sorted([k for k in f.keys() if k.startswith("episode_")])
        print(f"\nTotal episodes: {len(episode_keys)}")

        # Collect statistics
        all_positions = []
        all_grippers = []
        all_position_deltas = []

        episodes_to_check = min(num_episodes, len(episode_keys))
        print(f"\n=== Inspecting {episodes_to_check} episodes ===\n")

        for i, ep_key in enumerate(episode_keys[:episodes_to_check]):
            ep = f[ep_key]
            proprio = ep["proprio"][:]  # (T, 13)

            # Extract components: [R_flat(9), pos(3), gripper(1)]
            positions = proprio[:, 9:12]
            grippers = proprio[:, 12]
            position_deltas = np.diff(positions, axis=0)

            all_positions.append(positions)
            all_grippers.append(grippers)
            all_position_deltas.append(position_deltas)

            print(f"Episode {ep_key}: {len(proprio)} steps")
            print(
                f"  Pos X: [{positions[:, 0].min():.4f}, {positions[:, 0].max():.4f}]"
            )
            print(
                f"  Pos Y: [{positions[:, 1].min():.4f}, {positions[:, 1].max():.4f}]"
            )
            print(
                f"  Pos Z: [{positions[:, 2].min():.4f}, {positions[:, 2].max():.4f}]"
            )

        # Aggregate
        all_positions = np.concatenate(all_positions, axis=0)
        all_grippers = np.concatenate(all_grippers, axis=0)
        all_position_deltas = np.concatenate(all_position_deltas, axis=0)

        analyze_positions(all_positions, all_grippers, all_position_deltas)


def inspect_streaming(dataset: str, num_windows: int = 100, data_dir: str = None):
    """Inspect streaming RTX data to understand position scales."""
    from imitation.data.rtx_stream_loader import RTXStreamLoader

    print(f"\n{'=' * 60}")
    print(f"Inspecting STREAMING: {dataset}")
    print(f"{'=' * 60}\n")

    loader = RTXStreamLoader(
        dataset_name=dataset,
        split="train",
        window_size=8,
        loss_horizon=1,
        image_size=(224, 224),
        data_dir=data_dir,
        repeat=False,
        use_subprocess=False,
        shuffle_buffer_size=0,
        shuffle_files=False,
    )

    # Collect statistics
    all_positions = []
    all_grippers = []
    all_position_deltas = []
    all_twists = []

    print(f"Collecting data from {num_windows} windows...\n")

    for i, batch in enumerate(loader):
        if i >= num_windows:
            break

        if i % 20 == 0:
            print(f"  Window {i}/{num_windows}...")

        # proprio: (W, 13), actions: (H, 7)
        proprio = batch["proprio"].numpy()  # (W, 13)
        actions = batch["actions"].numpy()  # (H, 7)

        # Extract positions: [R_flat(9), pos(3), gripper(1)]
        positions = proprio[:, 9:12]  # (W, 3)
        grippers = proprio[:, 12]  # (W,)
        position_deltas = np.diff(positions, axis=0)  # (W-1, 3)

        all_positions.append(positions)
        all_grippers.append(grippers)
        all_position_deltas.append(position_deltas)
        all_twists.append(actions)

        # Print first few windows in detail
        if i < 3:
            print(f"\n  Window {i}:")
            print(f"    Proprio shape: {proprio.shape}")
            print(
                f"    First position:  [{positions[0, 0]:.6f}, {positions[0, 1]:.6f}, {positions[0, 2]:.6f}]"
            )
            print(
                f"    Last position:   [{positions[-1, 0]:.6f}, {positions[-1, 1]:.6f}, {positions[-1, 2]:.6f}]"
            )
            print(
                f"    Position delta:  {np.linalg.norm(positions[-1] - positions[0]):.6f}"
            )
            print(
                f"    Twist (action):  [{actions[0, 0]:.6f}, {actions[0, 1]:.6f}, {actions[0, 2]:.6f}, {actions[0, 3]:.6f}, {actions[0, 4]:.6f}, {actions[0, 5]:.6f}]"
            )

    if not all_positions:
        print("No data collected!")
        return

    # Aggregate
    all_positions = np.concatenate(all_positions, axis=0)
    all_grippers = np.concatenate(all_grippers, axis=0)
    all_position_deltas = np.concatenate(all_position_deltas, axis=0)
    all_twists = np.concatenate(all_twists, axis=0)

    print(f"\nCollected {len(all_positions)} position samples")

    analyze_positions(all_positions, all_grippers, all_position_deltas, all_twists)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect data to understand position scales"
    )

    # Mode selection
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use RTX streaming loader instead of local HDF5",
    )

    # HDF5 options
    parser.add_argument(
        "--data_path", type=str, default=None, help="Path to preprocessed HDF5 file"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to inspect (HDF5 mode)",
    )

    # Streaming options
    parser.add_argument(
        "--dataset",
        type=str,
        default="droid",
        help="Dataset name for streaming (default: droid)",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None, help="TFDS data directory"
    )
    parser.add_argument(
        "--num_windows",
        type=int,
        default=100,
        help="Number of windows to inspect (streaming mode)",
    )

    args = parser.parse_args()

    if args.streaming:
        inspect_streaming(args.dataset, args.num_windows, args.data_dir)
    elif args.data_path:
        if not os.path.exists(args.data_path):
            print(f"Error: File not found: {args.data_path}")
            return
        inspect_hdf5(args.data_path, args.num_episodes)
    else:
        print("Error: Specify either --streaming or --data_path")
        parser.print_help()


if __name__ == "__main__":
    main()
