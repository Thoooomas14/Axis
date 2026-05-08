"""
DROID Dataset Preprocessor

Streams DROID dataset from GCS and saves minimal episode data locally.
Stores only: images (one camera, resized), 13D poses, twists per timestep, and goal vectors.
Utilizes a Producer-Consumer multi-threading architecture to balance I/O bounds with GPU/Disk bounds.

Usage:
    python -m imitation.data.preprocessor --output E:/data/droid_processed/ --workers 8
"""

import os
import json
import tempfile
import sys
import argparse
import shutil
import concurrent.futures
import multiprocessing
from pathlib import Path
import torch
import numpy as np
import cv2
from tqdm import tqdm
import pypose as pp
from requests import adapters
from scipy.spatial.transform import Rotation as R
from google.cloud import storage
from google.cloud.exceptions import NotFound
import imageio
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from imitation.utils.logger import TqdmLoggingHandler

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

tqdm_handler = TqdmLoggingHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
tqdm_handler.setFormatter(formatter)
logger.addHandler(tqdm_handler)

from imitation.data.goal_oracle import GoalOracle  # noqa: E402
from imitation.data.utils.droid_utils import TrajectoryReader, MP4Reader  # noqa: E402


def get_free_space_gb(path: Path) -> float:
    """Get free disk space in GB for the drive containing path."""
    try:
        total, used, free = shutil.disk_usage(path.parent)
        return free / (1024**3)
    except Exception:
        return float("inf")


def cleanup_orphaned_episodes(output_dir):
    """Scans the output directory and deletes any incomplete episodes from previous runs."""
    if not os.path.exists(output_dir):
        return

    logger.info("Running cleanup: Scanning for orphaned episodes...")
    orphans_deleted = 0

    for item in os.listdir(output_dir):
        ep_path = os.path.join(output_dir, item)
        # Only check directories that look like episode folders
        if os.path.isdir(ep_path) and item.startswith("episode_"):
            # If the folder exists, but the goal vector doesn't, it's an orphan!
            if not os.path.exists(os.path.join(ep_path, "goal.npy")):
                shutil.rmtree(ep_path, ignore_errors=True)
                orphans_deleted += 1

    if orphans_deleted > 0:
        logger.info(f"Cleanup complete: Deleted {orphans_deleted} orphaned episodes.")
    else:
        logger.info("Cleanup complete: No orphaned episodes found.")


DEFAULT_GRIPPER = torch.tensor([0.0])


def get_pose(obs) -> torch.Tensor:
    if "cartesian_position" in obs:
        p = obs["cartesian_position"]
        pos = torch.as_tensor(p[:3])
        rotation = pp.euler2SO3(p[3:])
    else:
        pos = torch.zeros(3)
        rotation = pp.identity_SO3()

    g = DEFAULT_GRIPPER
    if "gripper_position" in obs:
        g = torch.as_tensor(obs["gripper_position"]).reshape(-1)[:1]

    rot_flat = rotation.matrix().reshape(-1)
    return torch.cat([rot_flat, pos, g])


def get_action(obs) -> np.ndarray:
    """
    Extracts a flat 7D action array (6D cartesian velocity + 1D gripper)
    from the raw HDF5 action dictionary.
    """
    if isinstance(obs, dict):
        # Extract cartesian and gripper values (default to zeros if missing)
        cart = obs.get("cartesian_velocity", np.zeros(6))
        grip = obs.get("gripper_velocity", np.zeros(1))

        # Flatten them
        cart = np.array(cart).flatten()
        grip = np.array(grip).flatten()

        # Combine into a single array
        full_action = np.concatenate([cart, grip])

        # Pad with zeros if it's too short, or truncate if it's too long to guarantee 7D
        if len(full_action) < 7:
            full_action = np.pad(full_action, (0, 7 - len(full_action)))
        return full_action[:7]
    return np.zeros(7)


def process_image(img, target_size=224) -> tuple[np.ndarray | None, tuple[int, int]]:
    if img is None:
        return None, (0, 0)

    if hasattr(img, "numpy"):
        img = img.numpy()

    w, h = img.shape[:2]
    min_dim = min(w, h)
    pad_w = w - min_dim
    pad_h = h - min_dim
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)  # Ensure exact padding if odd
    left, right = pad_w // 2, pad_w - (pad_w // 2)

    # FAST OpeCV C++ Implementation
    padded_image = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(1, 1, 1)
    )

    return cv2.resize(
        padded_image, (target_size, target_size), interpolation=cv2.INTER_AREA
    ), (h, w)


def get_extrinsics(
    extrinsic_superset_data, extrinsic_data, target_episode, target_serial
):
    if target_episode in extrinsic_superset_data:
        episode_data = extrinsic_superset_data[target_episode]
    elif target_episode in extrinsic_data:
        episode_data = extrinsic_data[target_episode]
    else:
        return None

    if target_serial not in episode_data:
        return None

    raw_values = episode_data[target_serial]
    pos = np.array(raw_values[0:3])
    rot_matrix = R.from_euler("xyz", raw_values[3:6]).as_matrix()
    t_cam_to_base = np.eye(4)
    t_cam_to_base[:3, :3] = rot_matrix
    t_cam_to_base[:3, 3] = pos

    return np.linalg.inv(t_cam_to_base)  # invert matrix to return base to cam


def project_trajectory_to_image(
    ee_positions, intrinsics, extrinsics_base2cam, distortion=None
):
    """Projects a 3D Base-Frame coordinate into 2D Image Pixel coordinates."""
    ee_positions = np.atleast_2d(ee_positions).astype(np.float64)
    R_matrix = extrinsics_base2cam[:3, :3]
    t_vec = extrinsics_base2cam[:3, 3]
    r_vec, _ = cv2.Rodrigues(R_matrix)

    if distortion is None:
        distortion = np.zeros((5, 1), dtype=np.float64)

    image_points, _ = cv2.projectPoints(
        ee_positions, r_vec, t_vec, intrinsics, distortion
    )
    return image_points.reshape(-1, 2)


worker_client = None
worker_bucket = None
worker_stop_event = None  # Add this global


def init_worker_process(project_id, bucket_name, stop_event):
    """Initializes GCS client locally and attaches the stop event."""
    global worker_client, worker_bucket, worker_stop_event

    worker_stop_event = (
        stop_event  # Store the shared event globally in the child process
    )

    from google.cloud import storage
    from requests import adapters

    worker_client = storage.Client(project=project_id)
    adapter = adapters.HTTPAdapter(pool_connections=5, pool_maxsize=5)
    worker_client._http.mount("https://", adapter)
    worker_client._http.mount("http://", adapter)
    worker_bucket = worker_client.bucket(bucket_name, user_project=project_id)


def extract_episode_worker(
    blob_name,
    episode_id,
    valid_instruction,
    target_serial,
    T_base2cam,
    intrinsics_matrix,
    output_dir,
):
    """
    PRODUCER FUNCTION: Downloads data, decodes video, resizes images, and extracts trajectory arrays.
    """
    global worker_bucket  # Use the locally initialized bucket!
    global worker_stop_event  # Use the shared stop event to check for shutdown signals!

    if worker_stop_event is not None and worker_stop_event.is_set():
        return None

    temp_dir = tempfile.mkdtemp(dir="/dev/shm")
    base_directory = blob_name.rsplit("/", 1)[0] + "/"

    try:
        assert worker_bucket is not None, "Worker bucket not initialized!"  # noqa: S101
        # Download files using the worker's local bucket
        h5_blob = worker_bucket.blob(blob_name=f"{base_directory}trajectory.h5")
        mp4_blob = worker_bucket.blob(
            f"{base_directory}recordings/MP4/{target_serial}.mp4"
        )

        local_h5_path = os.path.join(temp_dir, f"episode_{episode_id}.h5")
        local_mp4_path = os.path.join(temp_dir, f"episode_{episode_id}.mp4")

        h5_blob.download_to_filename(local_h5_path)
        mp4_blob.download_to_filename(local_mp4_path)

        # Extract Episode Trajectory and Video
        traj_reader = TrajectoryReader(local_h5_path, read_images=False)
        horizon = traj_reader.length()

        if horizon is None or horizon <= 0:
            logger.warning(
                f"Skipping episode {episode_id}: Invalid horizon length ({horizon})"
            )
            traj_reader.close()
            return None

        mp4_reader = MP4Reader(local_mp4_path, target_serial)
        mp4_reader.set_reading_parameters(image=True, concatenate_images=False)

        images = []
        poses = []
        twists = []
        orig_shape = None

        for i in range(horizon):
            # POISON PILL CHECK: If main thread says stop, abort immediately.
            assert worker_stop_event is not None, "Worker stop event not initialized!"  # noqa: S101
            if worker_stop_event.is_set():
                traj_reader.close()
                mp4_reader.disable_camera()
                return None

            step = traj_reader.read_timestep(index=i)
            obs = step.get("action", step)

            # Poses
            pose_tensor = get_pose(obs)
            poses.append(
                pose_tensor.numpy() if hasattr(pose_tensor, "numpy") else pose_tensor
            )

            # Twists/Actions
            action_array = get_action(obs)
            twists.append(action_array)

            # Image Frame
            cam_data = mp4_reader.read_camera()
            if cam_data is None or "image" not in cam_data:
                img = images[-1] if images else np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                raw_img = list(cam_data["image"].values())[0]
                img, current_shape = process_image(raw_img, target_size=224)
                if orig_shape is None:
                    orig_shape = current_shape
            images.append(img)

        traj_reader.close()
        mp4_reader.disable_camera()

        images = np.array(images, dtype=np.uint8)
        poses = np.array(poses, dtype=np.float32)
        twists = np.array(twists, dtype=np.float32)

        velocity_eps = 1e-4
        gripper_eps = 1e-3

        # Calculate L2 norms of the 6D Cartesian twist for each timestep
        cartesian_norms = np.linalg.norm(twists[:, :6], axis=1)
        gripper_deltas = np.abs(twists[:, 6])

        # Create a boolean mask of active frames
        active_mask = (cartesian_norms > velocity_eps) | (gripper_deltas > gripper_eps)

        # Find the first index where movement exceeds our thresholds
        if np.any(active_mask):
            start_idx = np.argmax(active_mask)
        else:
            start_idx = 0  # Fallback if the entire episode is mathematically static

        # Apply the slice to remove the static prefix
        if start_idx > 0:
            images = images[start_idx:]
            poses = poses[start_idx:]
            twists = twists[start_idx:]
            horizon = len(images)

        # Calculate trajectory length
        if len(poses) > 1:
            positions = poses[:, 9:12]
            trajectory_length_m = float(
                np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
            )
        else:
            trajectory_length_m = 0.0

        # Use the operator's absolute command signal instead of physical state
        gripper_commands = twists[:, 6]

        # Binarize the commands (Threshold at 0.5 to separate intent to close vs open)
        is_closed = gripper_commands > 0.5

        # np.diff will find the exact frame where the boolean flips:
        # 0 -> 1 (Open -> Closed) equals 1
        # 1 -> 0 (Closed -> Open) equals -1
        transitions = np.diff(is_closed.astype(int))

        # The physical state reaches the new intent on the frame AFTER the diff
        starts_closing = np.where(transitions == 1)[0] + 1
        starts_opening = np.where(transitions == -1)[0] + 1

        # Grab the first close and the last open
        t_grasp = starts_closing[0] if len(starts_closing) > 0 else None
        t_drop = starts_opening[-1] if len(starts_opening) > 0 else None

        init_obj_coord, final_obj_coord = None, None

        if orig_shape is not None and (t_grasp is not None or t_drop is not None):
            try:
                if intrinsics_matrix is not None:

                    def apply_padding_scale(coord):
                        orig_height, orig_width = orig_shape
                        min_dim = min(orig_height, orig_width)

                        top = (orig_height - min_dim) // 2
                        left = (orig_width - min_dim) // 2

                        padded_x = coord[0] + left
                        padded_y = coord[1] + top

                        scale = 224 / max(orig_height, orig_width)
                        return np.array(
                            [padded_x * scale, padded_y * scale], dtype=np.float32
                        )

                    if t_grasp is not None:
                        pos_grasp = poses[t_grasp, 9:12]
                        proj_grasp = project_trajectory_to_image(
                            pos_grasp, intrinsics_matrix, T_base2cam
                        )
                        init_obj_coord = apply_padding_scale(proj_grasp[0])
                        for c in init_obj_coord:
                            if c < 0 or c > 224:
                                logger.warning(
                                    f"Grasp coordinate out of bounds for {episode_id}: {init_obj_coord}"
                                )
                                return None

                    if t_drop is not None:
                        pos_drop = poses[t_drop, 9:12]
                        proj_drop = project_trajectory_to_image(
                            pos_drop, intrinsics_matrix, T_base2cam
                        )
                        final_obj_coord = apply_padding_scale(proj_drop[0])
                        for c in final_obj_coord:
                            if c < 0 or c > 224:
                                logger.warning(
                                    f"Drop coordinate out of bounds for {episode_id}: {final_obj_coord}"
                                )
                                return None

            except Exception as e:
                logger.warning(f"Failed to project coordinates for {episode_id}: {e}")

        ep_dir = os.path.join(output_dir, f"episode_{episode_id}")
        os.makedirs(ep_dir, exist_ok=True)

        np.save(os.path.join(ep_dir, "images.npy"), images)
        np.save(os.path.join(ep_dir, "poses.npy"), poses)
        np.save(os.path.join(ep_dir, "twists.npy"), twists)

        with open(os.path.join(ep_dir, "instruction.txt"), "w") as f:
            f.write(valid_instruction if valid_instruction else "")

        return {
            "episode_id": episode_id,
            "ep_dir": ep_dir,
            "first_image": images[0]
            if len(images) > 0
            else np.zeros((224, 224, 3), dtype=np.uint8),
            "instruction": valid_instruction,
            "target_serial": target_serial,
            "horizon": horizon,
            "trajectory_length_m": trajectory_length_m,
            "init_obj_coord": init_obj_coord,
            "final_obj_coord": final_obj_coord,
        }

    except NotFound:
        return None
    except Exception as e:
        logger.error(f"Unexpected error on {episode_id}: {e}")
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def get_high_concurrency_client(project_id, max_pool):
    client = storage.Client(project="axis-480014")
    adapter = adapters.HTTPAdapter(
        pool_connections=max_pool + 5, pool_maxsize=max_pool + 5
    )
    client._http.mount("https://", adapter)
    client._http.mount("http://", adapter)
    return client


def process_droid_raw_episodes(
    bucket_name="gresearch",
    prefix="robotics/droid_raw/1.0.1/",
    extrinsics_superset="imitation/data/cam2base_extrinsic_superset.json",
    extrinsics="imitation/data/cam2base_extrinsics.json",
    intrinsics="imitation/data/intrinsics.json",
    camera_serials="imitation/data/camera_serials.json",
    language_annotations="imitation/data/droid_language_annotations.json",
    max_workers=8,
    output_dir="./data_processed/",
    oracle: GoalOracle | None = None,
    min_free_space_gb=50.0,
    test_mode=False,
):

    client = get_high_concurrency_client("axis-480014", max_workers + 10)
    bucket = client.bucket(bucket_name, user_project="axis-480014")

    logger.info("Loading metadata files...")
    with open(extrinsics_superset, "rb") as f:
        extrinsics_superset_data = json.loads(f.read())
    with open(extrinsics, "rb") as f:
        extrinsics_data = json.loads(f.read())
    with open(intrinsics, "rb") as f:
        intrinsics_data = json.loads(f.read())
    with open(camera_serials, "rb") as f:
        serial_map = json.loads(f.read())
    with open(language_annotations, "rb") as f:
        annotations_data = json.loads(f.read())

    RED_FLAG = ["and", "or", "then"]

    logger.info("Scanning for valid episodes...")
    blobs = client.list_blobs(bucket, prefix=prefix, match_glob="**/*metadata_*.json")
    valid_blobs = (b for b in blobs if "success" in b.name)

    logger.info(f"Streaming episodes using ProcessPool (Workers={max_workers})...")
    all_metadata = []
    successful_episodes = 0
    pbar = tqdm(desc="Processed Episodes", unit=" episodes")

    global worker_stop_event
    manager = multiprocessing.Manager()
    worker_stop_event = manager.Event()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker_process,
        initargs=(
            "axis-480014",
            bucket_name,
            worker_stop_event,
        ),
    ) as executor:
        active_futures = {}
        blob_iterator = iter(valid_blobs)
        is_submitting = True

        # THE UNIFIED EVENT LOOP:
        # Keeps running as long as there are blobs to read OR tasks still processing
        while is_submitting or active_futures:
            # 1. THE PRODUCER: Fill the queue up to a healthy limit (e.g., 2x worker count)
            while is_submitting and len(active_futures) < (max_workers * 2):
                try:
                    blob = next(blob_iterator)
                except StopIteration:
                    is_submitting = False
                    break  # Reached the end of the GCS blobs

                filename = blob.name.split("/")[-1]
                episode_id = filename.replace("metadata_", "").replace(".json", "")

                # --- Validation Checks ---
                instructions = annotations_data.get(episode_id, {})
                if not instructions.get("language_instruction1"):
                    continue

                valid_instruction = None
                for _, instr in instructions.items():
                    clean_instr = instr.lower().replace(".", "").replace(",", "")
                    words = set(clean_instr.split())
                    if any(red_flag in words for red_flag in RED_FLAG):
                        continue
                    valid_instruction = instr
                    break

                if not valid_instruction:
                    continue

                episode_serials = serial_map.get(episode_id, {})
                target_serial = episode_serials.get(
                    "ext1_cam_serial"
                ) or episode_serials.get("ext2_cam_serial")
                if not target_serial:
                    continue

                T_base2cam = get_extrinsics(
                    extrinsics_superset_data, extrinsics_data, episode_id, target_serial
                )
                if T_base2cam is None:
                    continue

                intrinsics_matrix = None
                if (
                    episode_id in intrinsics_data
                    and target_serial in intrinsics_data[episode_id]
                ):
                    cam_info = intrinsics_data[episode_id][target_serial]
                    if (
                        "cameraMatrix" in cam_info
                        and len(cam_info["cameraMatrix"]) == 4
                    ):
                        fx, cx, fy, cy = cam_info["cameraMatrix"]
                        intrinsics_matrix = np.array(
                            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                            dtype=np.float64,
                        )

                # Submit to worker
                future = executor.submit(
                    extract_episode_worker,
                    blob.name,
                    episode_id,
                    valid_instruction,
                    target_serial,
                    T_base2cam,
                    intrinsics_matrix,
                    output_dir,
                )
                active_futures[future] = episode_id

            # 2. THE CONSUMER: Wait for at least one future to finish, then run GoalOracle
            if active_futures:
                done, _ = concurrent.futures.wait(
                    active_futures.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                for future in done:
                    # Pop the finished future from the tracking dictionary
                    episode_id = active_futures.pop(future)

                    free_space = get_free_space_gb(Path(output_dir))
                    if free_space < min_free_space_gb:
                        logger.warning(
                            f"Free disk space critically low ({free_space:.2f} GB)! Initiating shutdown..."
                        )
                        worker_stop_event.set()
                        for f in active_futures.keys():
                            f.cancel()
                        is_submitting = False
                        active_futures.clear()  # Force loop exit
                        break

                    result = future.result()
                    if result is None:
                        continue

                    # Run Goal Oracle
                    goal_vector = None
                    if oracle is not None:
                        goal_tensor = oracle.encode_goal(
                            init_obj_coordinate=result["init_obj_coord"],
                            final_obj_coordinate=result["final_obj_coord"],
                            img=result["first_image"],
                            instruction=result["instruction"],
                        )
                        if goal_tensor is not None:
                            goal_vector = goal_tensor.cpu().numpy()

                    if goal_vector is None:
                        logger.warning(
                            f"Skipping episode {episode_id}: Goal vector generation failed."
                        )
                        shutil.rmtree(result["ep_dir"], ignore_errors=True)
                        pbar.update(1)
                        continue

                    # Main thread saves the goal vector
                    np.save(os.path.join(result["ep_dir"], "goal.npy"), goal_vector)

                    # Compile Metadata
                    all_metadata.append(
                        {
                            "episode_id": episode_id,
                            "local_path": result["ep_dir"],
                            "episode_length": result["horizon"],
                            "trajectory_length_m": result["trajectory_length_m"],
                            "instruction": result["instruction"],
                            "camera_serial": result["target_serial"],
                            "has_goal": True,
                        }
                    )

                    successful_episodes += 1
                    pbar.update(1)
                    pbar.set_postfix({"success": successful_episodes})

                    if test_mode:
                        gif_path = os.path.join(result["ep_dir"], "goal_overlay.gif")
                        sample_txt_path = os.path.join(
                            result["ep_dir"], "sample_data.txt"
                        )

                        saved_images = np.load(
                            os.path.join(result["ep_dir"], "images.npy")
                        )
                        saved_poses = np.load(
                            os.path.join(result["ep_dir"], "poses.npy")
                        )
                        saved_twists = np.load(
                            os.path.join(result["ep_dir"], "twists.npy")
                        )

                        with open(sample_txt_path, "w") as f:
                            f.write(
                                f"{'=' * 50}\nEPISODE {episode_id} DATA SAMPLE\n{'=' * 50}\n"
                            )
                            f.write(
                                f"First 100 Poses (Array shape: {saved_poses.shape}):\n"
                            )
                            with np.printoptions(precision=4, suppress=True):
                                f.write(f"{saved_poses[:100]}\n\n")
                            f.write(
                                f"First 100 Twists/Actions (Array shape: {saved_twists.shape}):\n"
                            )
                            with np.printoptions(precision=4, suppress=True):
                                f.write(f"{saved_twists[:100]}\n")
                            f.write(f"{'=' * 50}\n")

                        logger.info(f"Sample data saved to {sample_txt_path}")
                        generate_test_gif(
                            saved_images,
                            goal_vector,
                            gif_path,
                            result["init_obj_coord"],
                            result["final_obj_coord"],
                        )

                        if successful_episodes >= 5:
                            logger.info(
                                "Test mode completed 5 episodes. Shutting down workers..."
                            )
                            worker_stop_event.set()
                            for f in active_futures.keys():
                                f.cancel()
                            is_submitting = False
                            active_futures.clear()  # Force loop exit
                            break

                    # Periodic metadata sync to disk
                    if len(all_metadata) % 50 == 0:
                        meta_path = os.path.join(output_dir, "dataset_metadata.json")
                        with open(meta_path, "w") as f:
                            json.dump(all_metadata, f, indent=4)

    pbar.close()

    total_episodes = len(all_metadata)

    if total_episodes > 0:
        total_frames = sum(ep["episode_length"] for ep in all_metadata)
        avg_length = total_frames / total_episodes
        total_distance = sum(ep["trajectory_length_m"] for ep in all_metadata)
        avg_distance_m = total_distance / total_episodes
        min_length = min(ep["episode_length"] for ep in all_metadata)
        max_length = max(ep["episode_length"] for ep in all_metadata)
    else:
        total_frames = avg_length = min_length = max_length = 0

    dataset_summary = {
        "dataset_summary": {
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "avg_episode_length": float(avg_length),
            "min_episode_length": min_length,
            "max_episode_length": max_length,
            "avg_trajectory_length_m": float(avg_distance_m),
            "reach_limit_m": 0.855,
            "norm_type": "workspace_linear",
        },
        "episodes": all_metadata,
    }

    # Final Meta Save
    meta_path = os.path.join(output_dir, "dataset_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(dataset_summary, f, indent=4)

    logger.info(f"Successfully finished preprocessing! Metadata saved to {meta_path}")


def preprocess_dataset(args):
    """Entry point for preprocess_dataset logic adapted to npy saving."""
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output Directory Prepared: {output_path}")

    oracle = None
    try:
        logger.info("Initializing GoalOracle Models in main thread...")
        oracle = GoalOracle()
    except Exception as e:
        logger.warning(f"Could not initialize GoalOracle: {e}")

    # Commence Generator Workflow
    process_droid_raw_episodes(
        bucket_name="gresearch",
        prefix="robotics/droid_raw/1.0.1/",
        extrinsics_superset="imitation/data/cam2base_extrinsic_superset.json",
        extrinsics="imitation/data/cam2base_extrinsics.json",
        intrinsics="imitation/data/intrinsics.json",
        camera_serials="imitation/data/camera_serials.json",
        language_annotations="imitation/data/droid_language_annotations.json",
        max_workers=args.workers,
        output_dir=str(output_path),
        oracle=oracle,
        min_free_space_gb=args.min_free_space_gb,
        test_mode=args.test,
    )


def generate_test_gif(images, goal_vector, output_path, init_coord, final_coord):
    """
    Overlays Goal Oracle bounding boxes on the episode frames and saves as a GIF.
    Blue = Initial Position, Red = Target Position.
    """
    gif_frames = []

    # FIX: Ensure coordinates are integer tuples for OpenCV
    if init_coord is None:
        init_coord = (0, 0)
    else:
        init_coord = (int(init_coord[0]), int(init_coord[1]))

    if final_coord is None:
        final_coord = (0, 0)
    else:
        final_coord = (int(final_coord[0]), int(final_coord[1]))

    # Extract and un-normalize bounding boxes (Goal Oracle normalizes by 224)
    # Format: [cx, cy, w, h]
    init_bb = goal_vector[3:7] * 224.0
    target_bb = goal_vector[7:11] * 224.0

    def get_rect(bb):
        cx, cy, w, h = bb
        return (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2))

    pt1_init, pt2_init = get_rect(init_bb)
    pt1_target, pt2_target = get_rect(target_bb)

    for img in images:
        frame = img.copy()  # Copy so we don't draw on the raw data

        # Draw Initial BB (Blue in BGR)
        cv2.rectangle(frame, pt1_init, pt2_init, (255, 0, 0), 2)
        cv2.putText(
            frame,
            "Init",
            (pt1_init[0], pt1_init[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
        )
        # This will now work successfully
        cv2.circle(frame, init_coord, 3, (255, 0, 0), -1)

        # Draw Target BB (Red in BGR)
        cv2.rectangle(frame, pt1_target, pt2_target, (0, 0, 255), 2)
        cv2.putText(
            frame,
            "Target",
            (pt1_target[0], pt1_target[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )

        # This will now work successfully
        cv2.circle(frame, final_coord, 3, (0, 0, 255), -1)

        # OpenCV reads video in BGR, but imageio expects RGB for GIFs
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gif_frames.append(frame_rgb)

    # Save the GIF at 10 Frames Per Second
    imageio.mimsave(output_path, gif_frames, fps=10)
    logger.info(f"Test GIF saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess robotics datasets to .npy chunks for local training"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory root to save structured processed trajectory files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Max workers for multi-thread processing. Keep this balanced to not overload RAM.",
    )
    parser.add_argument(
        "--min_free_space_gb",
        type=float,
        default=50.0,
        help="Minimum free space required (in GB) to continue processing.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode to only process a small subset of the dataset.",
    )
    args = parser.parse_args()
    try:
        preprocess_dataset(args)
    except KeyboardInterrupt:
        # Catch Ctrl+C and gracefully break the loop
        print()  # noqa: T201
        print("\n\n" + "=" * 60)  # noqa: T201
        logger.warning("[!] Keyboard Interrupt (Ctrl+C) detected!")
        logger.warning("Pipeline halted by user. Exiting immediately...")
        print("=" * 60 + "\n")  # noqa: T201
    except Exception as e:
        print()  # noqa: T201
        print("\n\n" + "=" * 60)  # noqa: T201
        logger.error(f"ERROR found: {e}")
        print("=" * 60 + "\n")  # noqa: T201
    finally:
        if worker_stop_event is not None:
            worker_stop_event.set()
        logger.info("Cleaning up resources and terminating threads...")

        # NUCLEAR OPTION: Kill any lingering zombie child processes
        import multiprocessing

        for child in multiprocessing.active_children():
            child.terminate()

        cleanup_orphaned_episodes(args.output)
        logger.info("Terminating process...")
        os._exit(0)
