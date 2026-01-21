"""
RTX Dataset Preprocessor

Streams RTX datasets (DROID, Fractal, etc.) from GCS and saves minimal episode data locally.
Stores only: images (one camera, resized) and 7D poses per timestep.
All derived computations (windowing, goals, twists) are done at load time by LocalDataLoader.

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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import h5py
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


def get_free_space_gb(path: Path) -> float:
    """Get free disk space in GB for the drive containing path."""
    try:
        total, used, free = shutil.disk_usage(path.parent)
        return free / (1024 ** 3)
    except Exception:
        return float('inf')  # If we can't check, assume infinite


def quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [qx, qy, qz, qw] to rotation vector."""
    quat = quat / (np.linalg.norm(quat) + 1e-8)
    return R.from_quat(quat).as_rotvec().astype(np.float32)


def get_pose(obs) -> np.ndarray:
    """Extract 7D pose: [rotvec (3), position (3), gripper (1)]."""
    pos = np.zeros(3, dtype=np.float32)
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    rotvec = None
    
    # Position and orientation
    if 'base_pose_tool_reached' in obs:
        p7 = obs['base_pose_tool_reached'].numpy()
        pos = p7[:3]
        quat = p7[3:7]
    elif 'cartesian_position' in obs:
        p6 = obs['cartesian_position'].numpy()
        if p6.shape[0] == 6:
            pos = p6[:3]
            euler = p6[3:]
            rotvec = R.from_euler('xyz', euler).as_rotvec().astype(np.float32)
    elif 'ee_pose' in obs:
        p7 = obs['ee_pose'].numpy()
        pos = p7[:3]
        if p7.shape[0] >= 7:
            quat = p7[3:7]
    
    if rotvec is None:
        rotvec = quat_to_rotvec(quat)
    
    # Gripper state
    g = 0.0
    if 'gripper_closed' in obs:
        g = obs['gripper_closed'].numpy()
    elif 'gripper_position' in obs:
        g = obs['gripper_position'].numpy()
    elif 'gripper_state' in obs:
        g = obs['gripper_state'].numpy()
    
    if not np.isscalar(g):
        g = g.item() if g.size == 1 else g[0]
    
    pose7 = np.zeros(7, dtype=np.float32)
    pose7[:3] = rotvec
    pose7[3:6] = pos
    pose7[6] = g
    return pose7


def process_image(img, size: tuple) -> np.ndarray:
    """Process image: resize and convert to CHW uint8."""
    if img is None:
        return np.zeros((3, size[0], size[1]), dtype=np.uint8)
    if hasattr(img, 'numpy'):
        img = img.numpy()
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return np.transpose(img, (2, 0, 1))  # HWC -> CHW


def preprocess_dataset(args):
    """Stream dataset and save preprocessed episodes to HDF5."""
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.config.set_visible_devices([], 'GPU')
    
    # Limit TensorFlow memory usage
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    import tensorflow_datasets as tfds
    
    # Validate output path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Output path: {output_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Data dir: {args.data_dir or 'GCS (default)'}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    
    # Always use 'train' split - train/val splitting is done at load time
    split = 'train'
    
    # Build dataset with minimal caching/prefetching
    read_config = tfds.ReadConfig(
        try_autocache=False, 
        add_tfds_id=False,
        interleave_cycle_length=1,  # Single stream (network bottleneck)
        interleave_block_length=1,
    )
    
    # GCS paths for Open X-Embodiment datasets (not in standard TFDS registry)
    gcs_base = args.data_dir if args.data_dir else "gs://gresearch/robotics"
    
    if args.dataset == 'droid':
        # DROID uses version 1.0.1
        full_path = f"{gcs_base}/droid/1.0.1"
        print(f"Loading DROID from: {full_path}")
        builder = tfds.builder_from_directory(builder_dir=full_path)
    elif 'fractal' in args.dataset:
        # Fractal uses version 0.1.0
        full_path = f"{gcs_base}/{args.dataset}/0.1.0"
        print(f"Loading Fractal from: {full_path}")
        builder = tfds.builder_from_directory(builder_dir=full_path)
    else:
        # Try standard TFDS registry for other datasets
        builder = tfds.builder(args.dataset, data_dir=args.data_dir, try_gcs=True)
    
    ds = builder.as_dataset(split=split, shuffle_files=False, read_config=read_config)
    
    # Get total episodes for progress bar
    total_episodes = builder.info.splits[split].num_examples
    if args.max_episodes > 0:
        total_episodes = min(total_episodes, args.max_episodes)
    
    print(f"Total episodes to process: {total_episodes}")
    
    # Image keys to try (in order)
    fallback_keys = ['image', 'exterior_image_1_left', 'wrist_image_left', 'exterior_image_2_left']
    img_keys = [args.image_key] if args.image_key else fallback_keys
    image_size = (args.image_size, args.image_size)
    
    # Metadata collection
    episode_lengths = []
    
    # Check for resume
    resume_from = 0
    skip_count = 0
    file_mode = 'w'
    
    if output_path.exists() and not args.overwrite:
        try:
            with h5py.File(output_path, 'r') as existing:
                # Count existing episodes
                episode_keys = [k for k in existing.keys() if k.startswith('episode_')]
                resume_from = len(episode_keys)
                
                if resume_from > 0:
                    print(f"\nResuming from existing file with {resume_from} episodes")
                    # Load existing episode lengths for metadata
                    for key in sorted(episode_keys):
                        if 'length' in existing[key].attrs:
                            episode_lengths.append(existing[key].attrs['length'])
                    
                    file_mode = 'a'  # Append mode
                    # We need to skip episodes in the dataset
                    # Account for skipped short episodes (estimate ~0.5% skip rate from user's run)
                    skip_count = resume_from + int(resume_from * 0.006)
                    print(f"Will skip ~{skip_count} episodes from dataset to resume")
        except Exception as e:
            print(f"Could not read existing file: {e}")
            print("Starting fresh...")
            file_mode = 'w'
            resume_from = 0
            episode_lengths = []
    
    # Open HDF5 file
    with h5py.File(output_path, file_mode) as f:
        # Store metadata (only on fresh start)
        if file_mode == 'w':
            f.attrs['dataset'] = args.dataset
            f.attrs['data_dir'] = args.data_dir or 'gcs'
            f.attrs['image_size'] = args.image_size
            f.attrs['image_key'] = args.image_key or 'auto'
        
        episode_idx = resume_from
        skipped = 0
        ds_episode_idx = 0
        
        pbar = tqdm(ds, total=total_episodes, desc="Processing episodes", initial=resume_from)
        
        for episode in pbar:
            if args.max_episodes > 0 and episode_idx >= args.max_episodes:
                break
            
            # Skip already-processed episodes when resuming
            ds_episode_idx += 1
            if ds_episode_idx <= skip_count:
                if ds_episode_idx % 1000 == 0:
                    pbar.set_description(f"Skipping to resume point ({ds_episode_idx}/{skip_count})")
                continue
            
            # Extract episode data
            imgs = []
            props = []
            
            for step in episode['steps']:
                obs = step['observation']
                
                # Find image
                img = None
                for key in img_keys:
                    if key in obs:
                        img = obs[key]
                        break
                
                imgs.append(process_image(img, image_size))
                props.append(get_pose(obs))
            
            # Skip very short episodes
            if len(imgs) < args.min_episode_length:
                skipped += 1
                continue
            
            # Stack arrays
            episode_lengths.append(len(imgs))  # Track for metadata
            imgs = np.array(imgs, dtype=np.uint8)    # (T, 3, H, W)
            props = np.array(props, dtype=np.float32) # (T, 7)
            
            # Check disk space before writing (minimum 2GB buffer)
            free_space = get_free_space_gb(output_path)
            if free_space < args.min_free_space_gb:
                print(f"\n\n{'='*60}")
                print(f"WARNING: Disk space low ({free_space:.2f} GB < {args.min_free_space_gb} GB)")
                print(f"Stopping gracefully to prevent file corruption.")
                print(f"{'='*60}\n")
                # Remove the last episode length since we didn't write it
                episode_lengths.pop()
                break
            
            # Try to write episode - catch disk full errors
            try:
                # Create episode group
                ep_group = f.create_group(f'episode_{episode_idx:06d}')
                
                # Save with compression
                ep_group.create_dataset('images', data=imgs, compression='gzip', compression_opts=4)
                ep_group.create_dataset('proprio', data=props, compression='gzip', compression_opts=4)
                ep_group.attrs['length'] = len(imgs)
                
                # Flush to disk periodically to ensure data is saved
                if episode_idx % 100 == 0:
                    f.flush()
                    
            except OSError as e:
                print(f"\n\n{'='*60}")
                print(f"DISK WRITE ERROR: {e}")
                print(f"Stopping gracefully to prevent file corruption.")
                print(f"{'='*60}\n")
                # Remove the last episode length since write failed
                episode_lengths.pop()
                # Try to remove the partially written group
                try:
                    del f[f'episode_{episode_idx:06d}']
                except:
                    pass
                break
            
            episode_idx += 1
            pbar.set_postfix({'saved': episode_idx, 'skipped': skipped, 'free_gb': f'{free_space:.1f}'})
            
            # Aggressive memory cleanup after each episode
            del imgs, props
            gc.collect()
            
            # TF cleanup every 10 episodes to prevent memory buildup
            if episode_idx % 10 == 0:
                tf.keras.backend.clear_session()
                gc.collect()
        
        # Store comprehensive metadata for training
        f.attrs['num_episodes'] = episode_idx
        f.attrs['total_frames'] = int(sum(episode_lengths))
        f.attrs['avg_episode_length'] = float(np.mean(episode_lengths)) if episode_lengths else 0.0
        f.attrs['min_episode_length'] = int(min(episode_lengths)) if episode_lengths else 0
        f.attrs['max_episode_length'] = int(max(episode_lengths)) if episode_lengths else 0
        
        # Final flush to ensure everything is written
        f.flush()
    
    print(f"\nDone! Saved {episode_idx} episodes to {output_path}")
    print(f"Skipped {skipped} short episodes (< {args.min_episode_length} frames)")
    print(f"Total frames: {sum(episode_lengths):,}")
    print(f"Avg episode length: {np.mean(episode_lengths):.1f} frames")
    
    # Print file size
    size_gb = output_path.stat().st_size / (1024**3)
    print(f"File size: {size_gb:.2f} GB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess RTX datasets (DROID, Fractal) for local training')
    
    # Required
    parser.add_argument('--output', type=str, required=True,
                        help='Output HDF5 file path (e.g., E:/data/droid.h5)')
    
    # Dataset source
    parser.add_argument('--dataset', type=str, default='droid',
                        help='Dataset name: droid, fractal20220817_data, etc.')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='TFDS data directory. For Fractal: gs://gresearch/robotics')
    
    # Processing options
    parser.add_argument('--image_key', type=str, default=None,
                        help='Specific image key (e.g., exterior_image_1_left for DROID)')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Output image size, square (default: 128)')
    parser.add_argument('--min_episode_length', type=int, default=20,
                        help='Skip episodes shorter than this (default: 20)')
    
    # Limits
    parser.add_argument('--max_episodes', type=int, default=0,
                        help='Maximum episodes to process, 0 = all')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing file instead of resuming')
    parser.add_argument('--min_free_space_gb', type=float, default=2.0,
                        help='Stop if disk space falls below this (GB, default: 2.0)')
    
    args = parser.parse_args()
    preprocess_dataset(args)
