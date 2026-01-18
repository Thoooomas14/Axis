"""
Subprocess-based DROID data loader.

Isolates TensorFlow in a child process to guarantee memory release.
When the child process dies, ALL its memory is released by the OS.
"""

import os
# Suppress TF logs globally for this module and its subprocesses
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import multiprocessing as mp
import numpy as np
import torch
from torch.utils.data import IterableDataset
import queue
import gc


def _episode_worker(data_dir, dataset_name, split, image_key, image_size, window_size, loss_horizon, result_queue, stop_event):
    """
    Worker process that loads episodes and sends numpy data back via queue.
    
    This runs in a SEPARATE PROCESS, so when it exits, all TensorFlow memory is released.
    """
    # Suppress TensorFlow logging BEFORE importing
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF logs
    os.environ['CUDA_VISIBLE_DEVICES'] = ''   # No GPU for worker
    
    import warnings
    warnings.filterwarnings('ignore')  # Suppress Python warnings
    
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')  # Only show errors
    tf.config.set_visible_devices([], 'GPU')
    
    import tensorflow_datasets as tfds
    import cv2
    from scipy.spatial.transform import Rotation as R
    
    # Import rotation utils
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.utils.rotation_utils import se3_log
    
    def quat_to_rotvec(quat):
        quat = quat / (np.linalg.norm(quat) + 1e-8)
        return R.from_quat(quat).as_rotvec().astype(np.float32)
    
    def process_image(img, size):
        if img is None:
            return np.zeros((3, size[0], size[1]), dtype=np.uint8)
        if hasattr(img, 'numpy'):
            img = img.numpy()
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        # Keep as uint8 [0, 255] for memory efficiency
        return np.transpose(img, (2, 0, 1))
    
    def get_pose(obs):
        pos = np.zeros(3, dtype=np.float32)
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        
        rotvec = None
        
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
    
    def compute_twist(pose_curr, pose_next):
        R_curr = R.from_rotvec(pose_curr[:3]).as_matrix()
        p_curr = pose_curr[3:6]
        T_curr = np.eye(4, dtype=np.float32)
        T_curr[:3, :3] = R_curr
        T_curr[:3, 3] = p_curr
        
        R_next = R.from_rotvec(pose_next[:3]).as_matrix()
        p_next = pose_next[3:6]
        T_next = np.eye(4, dtype=np.float32)
        T_next[:3, :3] = R_next
        T_next[:3, 3] = p_next
        
        T_curr_inv = np.eye(4, dtype=np.float32)
        T_curr_inv[:3, :3] = R_curr.T
        T_curr_inv[:3, 3] = -R_curr.T @ p_curr
        
        T_rel = T_curr_inv @ T_next
        
        T_rel_torch = torch.tensor(T_rel, dtype=torch.float32).unsqueeze(0)
        xi = se3_log(T_rel_torch).squeeze(0).numpy()
        
        gripper_delta = pose_next[6] - pose_curr[6]
        
        twist7 = np.zeros(7, dtype=np.float32)
        twist7[:6] = xi
        twist7[6] = gripper_delta
        return twist7
    
    # Build dataset
    try:
        read_config = tfds.ReadConfig(try_autocache=False, add_tfds_id=False)
        
        if data_dir and data_dir.startswith('gs://') and 'fractal' in dataset_name:
            full_path = f"{data_dir}/{dataset_name}/0.1.0"
            builder = tfds.builder_from_directory(builder_dir=full_path)
        else:
            builder = tfds.builder(dataset_name, data_dir=data_dir, try_gcs=True)
        
        ds = builder.as_dataset(split=split, shuffle_files=True, read_config=read_config)
        ds = ds.repeat()
        
        fallback_keys = ['image', 'exterior_image_1_left', 'wrist_image_left', 'exterior_image_2_left']
        img_keys = [image_key] if image_key else fallback_keys
        
        buffer_size = window_size + loss_horizon
        MAX_STEPS_PER_EPISODE = 200  # Limit to prevent OOM from huge episodes
        MAX_EPISODES_BEFORE_REFRESH = 100  # Refresh every 100 episodes
        PRESTART_AT = 50  # Start warming at 50 (50 eps ahead of refresh)
        episode_count = 0
        
        for episode in ds:
            if stop_event.is_set():
                break
            
            episode_count += 1
            
            # Extract episode data to numpy
            imgs = []
            props = []
            
            for s in episode['steps']:
                obs = s['observation']
                
                img = None
                for key in img_keys:
                    if key in obs:
                        img = obs[key]
                        break
                
                imgs.append(process_image(img, image_size))
                props.append(get_pose(obs))
            
            if len(imgs) < buffer_size:
                continue
            
            # If episode is too long, keep only the LAST N steps (task completion is at end)
            if len(imgs) > MAX_STEPS_PER_EPISODE:
                imgs = imgs[-MAX_STEPS_PER_EPISODE:]
                props = props[-MAX_STEPS_PER_EPISODE:]
            
            imgs = np.array(imgs, dtype=np.float32)
            props = np.array(props, dtype=np.float32)
            
            # Generate windows
            num_windows = len(imgs) - window_size - loss_horizon + 1
            
            for w in range(num_windows):
                if stop_event.is_set():
                    break
                
                w_imgs = imgs[w : w + window_size]
                w_props = props[w : w + window_size]
                
                # Compute twists
                target_twists = []
                target_poses = []
                start_idx = w + window_size - 1
                
                for h in range(loss_horizon):
                    curr_idx = start_idx + h
                    next_idx = curr_idx + 1
                    twist = compute_twist(props[curr_idx], props[next_idx])
                    target_twists.append(twist)
                    target_poses.append(props[next_idx])
                
                # Send window via queue (blocks if queue full)
                result_queue.put({
                    'images': w_imgs,
                    'proprio': w_props,
                    'actions': np.array(target_twists, dtype=np.float32),
                    'target_poses': np.array(target_poses, dtype=np.float32),
                })
            
            # Clear TF after each episode
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Signal main process to start warming up next worker
            if episode_count == PRESTART_AT:
                result_queue.put({'prestart': True})
            
            # Proactive refresh - exit and let main process restart us
            if episode_count >= MAX_EPISODES_BEFORE_REFRESH:
                result_queue.put({'refresh': True})
                return  # Exit cleanly
            
        # This should never happen with ds.repeat()
        print("[Worker] WARNING: Episode loop ended unexpectedly!")
        result_queue.put({'error': 'Worker episode loop ended unexpectedly (repeat() failed?)'})
            
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[Worker] Exception: {error_msg}")
        result_queue.put({'error': error_msg})


class SubprocessDROIDLoader(IterableDataset):
    """
    Data loader that runs TensorFlow in a subprocess.
    
    When the subprocess is killed, ALL TensorFlow memory is released by the OS.
    """
    
    def __init__(self, data_dir, dataset_name='droid', split='train', 
                 image_key=None, image_size=(128, 128),
                 window_size=10, loss_horizon=1, queue_size=32):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.split = split
        self.image_key = image_key
        self.image_size = image_size
        self.window_size = window_size
        self.loss_horizon = loss_horizon
        self.queue_size = queue_size
        
        # Goal oracle (runs in main process)
        from .goal_oracle import GoalOracle
        self.oracle = GoalOracle(output_dim=64)
    
    def _start_worker(self, ctx, result_queue, stop_event):
        """Start a new worker process."""
        worker = ctx.Process(
            target=_episode_worker,
            args=(
                self.data_dir, self.dataset_name, self.split,
                self.image_key, self.image_size,
                self.window_size, self.loss_horizon,
                result_queue, stop_event
            )
        )
        worker.start()
        return worker
    
    def __iter__(self):
        # Create queue and event for communication
        ctx = mp.get_context('spawn')  # spawn for clean TF isolation
        result_queue = ctx.Queue(maxsize=self.queue_size)
        stop_event = ctx.Event()
        
        # Start worker process
        worker = self._start_worker(ctx, result_queue, stop_event)
        next_worker = None  # Pre-warmed worker for seamless transition
        next_queue = None
        next_stop_event = None
        
        restart_count = 0
        max_restarts = 100  # Allow many restarts for long training
        consecutive_timeouts = 0
        max_consecutive_timeouts = 90  # Force restart after 900s (15 mins) of no data
        
        try:
            while True:
                try:
                    data = result_queue.get(timeout=10.0)  # 10s timeout
                    consecutive_timeouts = 0  # Reset on success
                except queue.Empty:
                    consecutive_timeouts += 1
                    
                    # Force restart if too many consecutive timeouts (stuck on I/O)
                    if consecutive_timeouts >= max_consecutive_timeouts:
                        print(f"[SubprocessLoader] Worker stuck ({consecutive_timeouts * 10}s). Force restarting...")
                        worker.terminate()
                        worker.join(timeout=2.0)
                        if worker.is_alive():
                            worker.kill()
                        consecutive_timeouts = 0
                        restart_count += 1
                        if restart_count > max_restarts:
                            print("[SubprocessLoader] Max restarts reached. Stopping.")
                            break
                        worker = self._start_worker(ctx, result_queue, stop_event)
                        continue
                    
                    # Check if worker died
                    if not worker.is_alive():
                        print(f"[SubprocessLoader] Worker died. Restarting... (attempt {restart_count + 1})")
                        restart_count += 1
                        if restart_count > max_restarts:
                            print("[SubprocessLoader] Max restarts reached. Stopping.")
                            break
                        worker = self._start_worker(ctx, result_queue, stop_event)
                    continue
                
                if 'error' in data:
                    print(f"[SubprocessLoader] Worker error: {data['error']}. Restarting...", flush=True)
                    restart_count += 1
                    if restart_count > max_restarts:
                        raise RuntimeError(f"Worker error after max restarts: {data['error']}")
                    worker.terminate()
                    worker.join(timeout=2.0)
                    # Kill pre-warmed worker if exists
                    if next_worker is not None:
                        next_stop_event.set()
                        next_worker.terminate()
                        next_worker = None
                    worker = self._start_worker(ctx, result_queue, stop_event)
                    continue
                
                # Handle prestart request (start warming next worker)
                if 'prestart' in data:
                    if next_worker is None:
                        # Pre-warming next worker in background
                        try:
                            from tqdm import tqdm
                            tqdm.write(f"[SubprocessLoader] Pre-warming next worker...")
                        except ImportError:
                            print(f"[SubprocessLoader] Pre-warming next worker...")
                            
                        next_queue = ctx.Queue(maxsize=self.queue_size)
                        next_stop_event = ctx.Event()
                        next_worker = self._start_worker(ctx, next_queue, next_stop_event)
                    continue  # This is just a signal, not data
                
                # Handle refresh request (switch to pre-warmed worker)
                if 'refresh' in data:
                    worker.join(timeout=2.0)  # Wait for clean exit
                    if next_worker is not None:
                        # Switching to pre-warmed worker
                        try:
                            from tqdm import tqdm
                            tqdm.write(f"[SubprocessLoader] Switching to pre-warmed worker")
                        except ImportError:
                            print(f"[SubprocessLoader] Switching to pre-warmed worker")
                            
                        worker = next_worker
                        result_queue = next_queue
                        stop_event = next_stop_event
                        next_worker = None
                        next_queue = None
                        next_stop_event = None
                    else:
                        # Cold start fallback
                        try:
                            from tqdm import tqdm
                            tqdm.write(f"[SubprocessLoader] No pre-warmed worker, cold starting...")
                        except ImportError:
                            print(f"[SubprocessLoader] No pre-warmed worker, cold starting...")
                            
                        worker = self._start_worker(ctx, result_queue, stop_event)
                    continue
                
                # Add goal (computed in main process to keep oracle state)
                start_pose = data['proprio'][0]
                end_pose = data['proprio'][-1]
                task_type = 0
                if start_pose[6] < 0.5 and end_pose[6] > 0.5:
                    task_type = 1
                elif start_pose[6] > 0.5 and end_pose[6] < 0.5:
                    task_type = 2
                goal_emb = self.oracle.encode_goal(task_type, start_pose, end_pose)
                
                yield {
                    'images': torch.tensor(data['images'], dtype=torch.float32) / 255.0,
                    'proprio': torch.tensor(data['proprio'], dtype=torch.float32),
                    'goal': goal_emb,
                    'actions': torch.tensor(data['actions'], dtype=torch.float32),
                    'target_poses': torch.tensor(data['target_poses'], dtype=torch.float32),
                }
                
        finally:
            # Cleanup: signal worker to stop and kill it
            stop_event.set()
            worker.terminate()
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.kill()
