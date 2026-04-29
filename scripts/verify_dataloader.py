import torch
import sys
import os
import numpy as np
import tensorflow as tf
import argparse
import time
import shutil
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from imitation.data.rtx_stream_loader import RTXStreamLoader


def create_fake_episode(length=20):
    """Creates a fake episode with a gripper change in the middle."""
    steps = []

    # Phase 1: Open (0-9)
    # Phase 2: Closed (10-19)

    for i in range(length):
        gripper = 1.0 if i >= 10 else 0.0

        obs = {
            "image": tf.zeros((224, 224, 3), dtype=tf.uint8),  # Fake image
            "ee_pose": tf.constant(np.random.randn(6).astype(np.float32)),
            "gripper_closed": tf.constant(np.array([gripper], dtype=np.float32)),
        }

        step = {
            "observation": obs,
            "action": {
                "world_vector": tf.zeros(3),
                "rotation_delta": tf.zeros(3),
                "gripper_closedness_action": tf.zeros(1),
            },
        }
        steps.append(step)

    return {"steps": steps, "language_instruction": tf.constant("pick up the object")}


def verify_loader_logic():
    print("Verifying RTXStreamLoader Logic (Mock Data)...")

    # Instantiate loader (don't iterate it directly to avoid GCS)
    loader = RTXStreamLoader(dataset_name="dummy", window_size=8)

    print("Creating fake episode (Length 20, Split at 10)...")
    episode = create_fake_episode(20)

    print("Processing episode...")
    windows = list(loader._process_episode(episode))

    print(f"Generated {len(windows)} windows.")

    # We expect 2 subtasks.
    # Subtask 1: 0-10 (Length 10). Windows: 10-8+1 = 3.
    # Subtask 2: 10-20 (Length 10). Windows: 10-8+1 = 3.
    # Total: 6 windows.

    if len(windows) > 0:
        batch = windows[0]
        print("\nSample Window:")
        print(f"  Images: {batch['images'].shape}")
        print(f"  Proprio: {batch['proprio'].shape}")
        print(f"  Goal: {batch['goal'].shape}")
        print(f"  Action: {batch['action'].shape}")
        print(f"  Requery: {batch['requery'].shape}")

        assert batch["images"].shape == (8, 3, 224, 224)
        assert batch["proprio"].shape == (8, 8)
        assert batch["goal"].shape == (64,)
        assert batch["action"].shape == (8,)
        assert batch["requery"].shape == (1,)

        # Check Requery Logic
        # The last window of Subtask 1 should have requery=1
        # Subtask 1 windows are indices 0, 1, 2.
        # Window 2 should have requery=1.

        print("\nChecking Requery Labels:")
        for i, w in enumerate(windows):
            req = w["requery"].item()
            print(f"  Window {i}: Requery={req}")

        # Expected: 0, 0, 1, 0, 0, 1
        expected = [0, 0, 1, 0, 0, 1]
        actual = [int(w["requery"].item()) for w in windows]

        if actual == expected:
            print("\n✅ Requery Logic Verified.")
        else:
            print(f"\n❌ Requery Logic Failed. Expected {expected}, got {actual}")

        print("\n✅ Verification Successful: Logic is correct.")

    else:
        print("\n❌ Verification Failed: No windows generated.")


def stress_test(args):
    print(f"Starting Stress Test for {args.dataset}...")
    print(f"Workers: {args.num_workers}, Shuffle Buffer: {args.shuffle_buffer_size}")

    # Check Shared Memory (common crash cause in Docker/Cloud)
    if os.path.exists("/dev/shm"):
        shm_total = shutil.disk_usage("/dev/shm").total / (1024**3)
        print(f"Shared Memory (/dev/shm) Total: {shm_total:.2f} GB_")
        if shm_total < 2.0 and args.num_workers > 0:
            print(
                "WARNING: Low shared memory detected (<2GB). Workers may crash. Try --num_workers 0."
            )

    loader = RTXStreamLoader(
        dataset_name=args.dataset,
        split="train",
        batch_size=1,
        window_size=8,
        shuffle_buffer_size=args.shuffle_buffer_size,
        data_dir=args.data_dir,
    )

    dataloader = torch.utils.data.DataLoader(
        loader,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("Iterating through DataLoader...")
    start_time = time.time()
    count = 0

    try:
        pbar = tqdm(total=args.steps)
        for i, batch in enumerate(dataloader):
            if i >= args.steps:
                break

            # Simulate basic tensor usage
            _ = batch["images"].float()

            count += 1
            pbar.update(1)

        print(
            f"\n✅ Stress Test Passed! Processed {count} batches in {time.time() - start_time:.2f}s"
        )

    except Exception as e:
        print(f"\n❌ Stress Test Failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="logic",
        choices=["logic", "stress"],
        help="Test mode",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fractal20220817_data",
        help="Dataset for stress test",
    )
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--shuffle_buffer_size", type=int, default=10, help="Shuffle buffer size"
    )
    parser.add_argument("--steps", type=int, default=1000, help="Steps to run")

    args = parser.parse_args()

    if args.mode == "logic":
        verify_loader_logic()
    else:
        stress_test(args)
