"""
DROID Dataset Preprocessor

Streams DROID dataset from GCS and saves minimal episode data locally.
Stores only: images (one camera, resized) and 13D poses per timestep.
All derived computations (windowing, goals, twists) are done at load time by LocalDataLoader.

13D Pose Format: [R_flat(9), pos(3), gripper(1)]

Usage:
    python -m imitation.data.preprocessor --output E:/data/droid.h5 --dataset droid
    python -m imitation.data.preprocessor --output E:/data/fractal.h5 --dataset fractal20220817_data
    python -m imitation.data.preprocessor --output ./test.h5 --dataset droid --max_episodes 10
"""

import os
import sys
import argparse
import gc
import shutil
from pathlib import Path

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import h5py
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ThreadPoolExecutor

import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from imitation.data.goal_oracle import extract_object_properties


def get_free_space_gb(path: Path) -> float:
    """Get free disk space in GB for the drive containing path."""
    try:
        total, used, free = shutil.disk_usage(path.parent)
        return free / (1024**3)
    except Exception:
        return float("inf")  # If we can't check, assume infinite


def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix."""
    quat = quat / (np.linalg.norm(quat) + 1e-8)
    return R.from_quat(quat).as_matrix().astype(np.float32)


def get_pose(obs) -> np.ndarray:
    """Extract 13D pose: [R_flat(9), position(3), gripper(1)]."""
    pos = np.zeros(3, dtype=np.float32)
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    rotmat = None

    # Position and orientation
    if "base_pose_tool_reached" in obs:
        p7 = obs["base_pose_tool_reached"].numpy()
        pos = p7[:3]
        quat = p7[3:7]
    elif "cartesian_position" in obs:
        p6 = obs["cartesian_position"].numpy()
        if p6.shape[0] == 6:
            pos = p6[:3]
            euler = p6[3:]
            rotmat = R.from_euler("xyz", euler).as_matrix().astype(np.float32)
    elif "ee_pose" in obs:
        p7 = obs["ee_pose"].numpy()
        pos = p7[:3]
        if p7.shape[0] >= 7:
            quat = p7[3:7]

    if rotmat is None:
        rotmat = quat_to_rotmat(quat)

    # Gripper state
    g = np.float32(0.0)
    if "gripper_closed" in obs:
        g = obs["gripper_closed"].numpy().astype(np.float32)
    elif "gripper_position" in obs:
        g = obs["gripper_position"].numpy().astype(np.float32)
    elif "gripper_state" in obs:
        g = obs["gripper_state"].numpy().astype(np.float32)

    if not np.isscalar(g):
        g = g.item()
        g = np.float32(g)

    # 13D pose: [R_flat(9), pos(3), gripper(1)]
    pose13 = np.zeros(13, dtype=np.float32)
    pose13[:9] = rotmat.flatten()
    pose13[9:12] = pos
    pose13[12] = g
    return pose13


def process_image(
    img, target_size=(224, 224)
) -> tuple[np.ndarray | None, tuple[int, int]]:
    """Crop and downsample images to target size."""
    if img is None:
        return None, (0, 0)

    # 1. Convert to numpy if necessary (handles TF/Torch tensors)
    if hasattr(img, "numpy"):
        img = img.numpy()

    # 2. Aspect Ratio Correct: Center Crop to Square
    h, w = img.shape[:2]
    if h != w:
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        img = img[top : top + min_dim, left : left + min_dim]

    # 3. High-Quality Resizing
    # INTER_AREA is mathematically superior for downsampling (prevents aliasing).
    if img.shape[0] != target_size[0] or img.shape[1] != target_size[1]:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # 4. CHW Transpose & Memory Efficiency
    # We return uint8 [0-255] here.
    # Normalization (/255.0) should happen on the GPU during training.
    processed_img = np.transpose(img, (2, 0, 1)).astype(np.uint8)  # (3, H, W)
    return processed_img, (h, w)


def preprocess_dataset(args):
    """Stream dataset and save preprocessed episodes to HDF5."""

    tf.get_logger().setLevel("ERROR")
    tf.config.set_visible_devices([], "GPU")

    # Limit TensorFlow memory usage
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    # Validate output path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(
        f"Output path: {output_path}\n"
        f"Dataset: {args.dataset}\n"
        f"Data dir: {args.data_dir or 'GCS (default)'}\n"
        f"Image size: {args.image_size}x{args.image_size}\n"
    )

    # Always use 'train' split - train/val splitting is done at load time
    split = "train"

    # Build dataset with minimal caching/prefetching
    read_config = tfds.ReadConfig(
        try_autocache=False,
        add_tfds_id=False,
        interleave_cycle_length=1,  # Single stream (network bottleneck)
        interleave_block_length=1,
    )

    # GCS paths for Open X-Embodiment datasets (not in standard TFDS registry)
    gcs_base = args.data_dir if args.data_dir else "gs://gresearch/robotics"

    if args.dataset == "droid":
        # DROID uses version 1.0.1
        full_path = f"{gcs_base}/droid/1.0.1"
        logging.info(f"Loading DROID from: {full_path}")
        builder = tfds.builder_from_directory(builder_dir=full_path)
    elif "fractal" in args.dataset:
        # Fractal uses version 0.1.0
        full_path = f"{gcs_base}/{args.dataset}/0.1.0"
        logging.info(f"Loading Fractal from: {full_path}")
        builder = tfds.builder_from_directory(builder_dir=full_path)
    else:
        # Try standard TFDS registry for other datasets
        builder = tfds.builder(args.dataset, data_dir=args.data_dir, try_gcs=True)

    ds = builder.as_dataset(split=split, shuffle_files=False, read_config=read_config)

    # Get total episodes for progress bar
    total_episodes = builder.info.splits[split].num_examples
    if args.max_episodes > 0:
        total_episodes = min(total_episodes, args.max_episodes)

    logging.info(f"Total episodes to process: {total_episodes}")

    # Image keys to try (in order)
    fallback_keys = ["exterior_image_1_left", "image", "exterior_image_2_left"]
    img_keys = [args.image_key] if args.image_key else fallback_keys
    image_size = (args.image_size, args.image_size)

    # Metadata collection
    episode_lengths = []

    # Check for resume
    resume_from = 0
    skip_count = 0
    file_mode = "w"

    if output_path.exists() and not args.overwrite:
        try:
            with h5py.File(output_path, "r") as existing:
                # Count existing episodes
                episode_keys = [k for k in existing.keys() if k.startswith("episode_")]
                resume_from = len(episode_keys)

                if resume_from > 0:
                    logging.info(
                        f"\nResuming from existing file with {resume_from} episodes"
                    )
                    # Load existing episode lengths for metadata
                    for key in sorted(episode_keys):
                        if "length" in existing[key].attrs:
                            episode_lengths.append(existing[key].attrs["length"])

                    file_mode = "a"  # Append mode
                    # We need to skip episodes in the dataset
                    # Account for skipped short episodes (estimate ~0.5% skip rate from user's run)
                    skip_count = resume_from + int(resume_from * 0.006)
                    logging.info(
                        f"Will skip ~{skip_count} episodes from dataset to resume"
                    )
        except Exception as e:
            logging.error(f"Could not read existing file: {e}")
            logging.info("Starting fresh...")
            file_mode = "w"
            resume_from = 0
            episode_lengths = []

    # Open HDF5 file
    with h5py.File(output_path, file_mode) as f:
        # Store metadata (only on fresh start)
        if file_mode == "w":
            f.attrs["dataset"] = args.dataset
            f.attrs["data_dir"] = args.data_dir or "gcs"
            f.attrs["image_size"] = args.image_size
            f.attrs["image_key"] = args.image_key or "auto"

        episode_idx = resume_from
        skipped = 0
        ds_episode_idx = 0
        episodes_with_instruction = 0

        pbar = tqdm(
            ds, total=total_episodes, desc="Processing episodes", initial=resume_from
        )

        executor = ThreadPoolExecutor(max_workers=6)

        for episode in pbar:
            if args.max_episodes > 0 and episode_idx >= args.max_episodes:
                break

            # Skip already-processed episodes when resuming
            ds_episode_idx += 1
            if ds_episode_idx <= skip_count:
                if ds_episode_idx % 1000 == 0:
                    pbar.set_description(
                        f"Skipping to resume point ({ds_episode_idx}/{skip_count})"
                    )
                continue

            # Extract episode data
            raw_imgs = []
            props = []
            language_instruction = ""

            raw_step_data = list(episode["steps"])

            for step in raw_step_data:
                obs = step["observation"]

                # Extract language instruction from first step (with fallback)
                if not language_instruction and "language_instruction" in step:
                    instr = step["language_instruction"]
                    if hasattr(instr, "numpy"):
                        instr = instr.numpy()
                    if isinstance(instr, bytes):
                        instr = instr.decode("utf-8", errors="ignore")
                    language_instruction = str(instr).strip() if instr else ""

                # Fallback: check observation/natural_language_instruction
                if not language_instruction and "natural_language_instruction" in obs:
                    instr = obs["natural_language_instruction"]
                    if hasattr(instr, "numpy"):
                        instr = instr.numpy()
                    if isinstance(instr, bytes):
                        instr = instr.decode("utf-8", errors="ignore")
                    language_instruction = str(instr).strip() if instr else ""

                # Find image key
                img_data = None
                for key in img_keys:
                    if key in obs:
                        img_data = obs[key]
                        break
                raw_imgs.append(img_data)
                props.append(get_pose(obs))

            img_results = list(
                executor.map(lambda x: process_image(x, image_size), raw_imgs)
            )

            processed_imgs = [res[0] for res in img_results]
            original_sizes = [res[1] for res in img_results]

            none_count = sum(1 for x in processed_imgs if x is None)
            if none_count > 5:
                skipped += 1
                pbar.set_postfix_str(f"Skipped: missing {none_count} images")
                continue

            processed_imgs = [
                x
                if x is not None
                else np.zeros((3, image_size[0], image_size[1]), dtype=np.uint8)
                for x in processed_imgs
            ]

            # Extract object properties from language instruction
            if language_instruction:
                episodes_with_instruction += 1

            object_props = extract_object_properties(language_instruction)
            object_props_vec = np.concatenate(
                [object_props["size"], object_props["color"], object_props["shape"]]
            ).astype(np.float32)  # (9,)

            # Skip very short episodes
            if len(processed_imgs) < args.min_episode_length:
                skipped += 1
                pbar.set_postfix_str(
                    f"Skipped: short episode ({len(processed_imgs)}/{args.min_episode_length})"
                )
                continue

            # Stack arrays
            episode_lengths.append(len(processed_imgs))  # Track for metadata
            imgs = np.array(processed_imgs, dtype=np.uint8)  # (T, 3, H, W)
            props = np.array(props, dtype=np.float32)  # (T, 13)
            episode_orig_size = original_sizes[0] if original_sizes else (0, 0)

            # Check disk space before writing (minimum 2GB buffer)
            free_space = get_free_space_gb(output_path)
            if free_space < args.min_free_space_gb:
                logging.warning(f"\n\n{'=' * 60}")
                logging.warning(
                    f"WARNING: Disk space low ({free_space:.2f} GB < {args.min_free_space_gb} GB)"
                )
                logging.warning("Stopping gracefully to prevent file corruption.")
                logging.warning(f"{'=' * 60}\n")
                # Remove the last episode length since we didn't write it
                episode_lengths.pop()
                break

            # Try to write episode - catch disk full errors
            try:
                # Create episode group
                ep_group = f.create_group(f"episode_{episode_idx:06d}")

                # Save with compression
                ep_group.create_dataset(
                    "images", data=imgs, compression="gzip", compression_opts=4
                )
                ep_group.create_dataset(
                    "proprio", data=props, compression="gzip", compression_opts=4
                )
                ep_group.create_dataset(
                    "object_props", data=object_props_vec
                )  # (9,) per episode
                ep_group.attrs["length"] = len(imgs)
                ep_group.attrs["language_instruction"] = language_instruction
                ep_group.attrs["has_instruction"] = bool(language_instruction)
                ep_group.attrs["orig_shape"] = (
                    episode_orig_size  # Store original size of first frame as representative
                )

                # Flush to disk periodically to ensure data is saved
                if episode_idx % 100 == 0:
                    f.flush()

            except OSError as e:
                logging.error(f"\n\n{'=' * 60}")
                logging.error(f"DISK WRITE ERROR: {e}")
                logging.error("Stopping gracefully to prevent file corruption.")
                logging.error(f"{'=' * 60}\n")
                # Remove the last episode length since write failed
                episode_lengths.pop()
                # Try to remove the partially written group
                f.pop(f"episode_{episode_idx:06d}", None)
                break

            episode_idx += 1
            pbar.set_postfix(
                {
                    "saved": episode_idx,
                    "skipped": skipped,
                    "instr": episodes_with_instruction,
                    "free_gb": f"{free_space:.1f}",
                }
            )

            # Aggressive memory cleanup after each episode
            del imgs, props
            gc.collect()

            # TF cleanup every 10 episodes to prevent memory buildup
            if episode_idx % 10 == 0:
                tf.keras.backend.clear_session()
                gc.collect()

        # Store comprehensive metadata for training
        f.attrs["num_episodes"] = episode_idx
        f.attrs["total_frames"] = int(sum(episode_lengths))
        f.attrs["avg_episode_length"] = (
            float(np.mean(episode_lengths)) if episode_lengths else 0.0
        )
        f.attrs["min_episode_length"] = (
            int(min(episode_lengths)) if episode_lengths else 0
        )
        f.attrs["max_episode_length"] = (
            int(max(episode_lengths)) if episode_lengths else 0
        )
        f.attrs["reach_limit_m"] = 0.855  # For Franka Panda in DROID
        f.attrs["norm_type"] = "workspace_linear"

        # Final flush to ensure everything is written
        f.flush()

    logging.info(f"\nDone! Saved {episode_idx} episodes to {output_path}")
    logging.info(
        f"Skipped {skipped} short episodes (< {args.min_episode_length} frames)"
    )
    logging.info(
        f"Episodes with language instruction: {episodes_with_instruction}/{episode_idx}"
        f" ({episodes_with_instruction / max(1, episode_idx) * 100:.1f}%)"
    )
    logging.info(f"Total frames: {sum(episode_lengths):,}")
    logging.info(f"Avg episode length: {np.mean(episode_lengths):.1f} frames")

    # Print file size
    size_gb = output_path.stat().st_size / (1024**3)
    logging.info(f"File size: {size_gb:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess  datasets for local training"
    )

    # Required
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output HDF5 file path (e.g., E:/data/droid.h5)",
    )

    # Dataset source
    parser.add_argument(
        "--dataset",
        type=str,
        default="droid",
        help="Dataset name: droid, fractal20220817_data, etc.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="TFDS data directory. For Fractal: gs://gresearch/robotics",
    )

    # Processing options
    parser.add_argument(
        "--image_key",
        type=str,
        default="exterior_image_1_left",
        help="Specific image key (e.g., exterior_image_1_left for DROID)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Output image size, square (default: 224)",
    )
    parser.add_argument(
        "--min_episode_length",
        type=int,
        default=20,
        help="Skip episodes shorter than this (default: 20)",
    )

    # Limits
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=0,
        help="Maximum episodes to process, 0 = all",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing file instead of resuming",
    )
    parser.add_argument(
        "--min_free_space_gb",
        type=float,
        default=2.0,
        help="Stop if disk space falls below this (GB, default: 2.0)",
    )

    args = parser.parse_args()
    preprocess_dataset(args)
