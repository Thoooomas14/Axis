"""
Unified RTX Dataset Loader with Subtask-Based Goals.

Supports both in-process and subprocess TensorFlow loading modes.
Subprocess mode isolates TF memory - when the child process dies, ALL its memory is released.
"""

import os
# Suppress TF logs globally
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
from torch.utils.data import IterableDataset
import numpy as np
from scipy.spatial.transform import Rotation as R
import gc
import multiprocessing as mp
import queue

# Import SE(3) utilities
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pypose as pp
from .goal_oracle import GoalOracle, extract_object_properties


def _episode_worker(data_dir, dataset_name, split, image_key, image_size, 
                    window_size, loss_horizon, result_queue, stop_event):
    """
    Worker process that loads episodes and sends numpy data back via queue.
    Runs in a SEPARATE PROCESS - when it exits, all TensorFlow memory is released.
    """
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    import warnings
    warnings.filterwarnings('ignore')
    
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.config.set_visible_devices([], 'GPU')
    
    import tensorflow_datasets as tfds
    import cv2
    from scipy.spatial.transform import Rotation as R
    
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    import pypose as pp
    
    def quat_to_rotmat(quat):
        """Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix."""
        quat = quat / (np.linalg.norm(quat) + 1e-8)
        return R.from_quat(quat).as_matrix().astype(np.float32)
    
    def process_image(img, size):
        if img is None:
            return np.zeros((3, size[0], size[1]), dtype=np.uint8)
        if hasattr(img, 'numpy'):
            img = img.numpy()
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        return np.transpose(img, (2, 0, 1))
    
    def get_pose(obs):
        """Extract 13D SE(3) pose: [R_flat(9), pos(3), gripper(1)]."""
        pos = np.zeros(3, dtype=np.float32)
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        rotmat = None
        
        if 'base_pose_tool_reached' in obs:
            p7 = obs['base_pose_tool_reached'].numpy()
            pos = p7[:3]
            quat = p7[3:7]
        elif 'cartesian_position' in obs:
            p6 = obs['cartesian_position'].numpy()
            if p6.shape[0] == 6:
                pos = p6[:3]
                euler = p6[3:]
                rotmat = R.from_euler('xyz', euler).as_matrix().astype(np.float32)
        elif 'ee_pose' in obs:
            p7 = obs['ee_pose'].numpy()
            pos = p7[:3]
            if p7.shape[0] >= 7:
                quat = p7[3:7]
        
        if rotmat is None:
            rotmat = quat_to_rotmat(quat)
        
        g = 0.0
        if 'gripper_closed' in obs:
            g = obs['gripper_closed'].numpy()
        elif 'gripper_position' in obs:
            g = obs['gripper_position'].numpy()
        elif 'gripper_state' in obs:
            g = obs['gripper_state'].numpy()
        
        if not np.isscalar(g):
            g = g.item() if g.size == 1 else g[0]
        
        # 13D pose: [R_flat(9), pos(3), gripper(1)]
        pose13 = np.zeros(13, dtype=np.float32)
        pose13[:9] = rotmat.flatten()  # Column-major flatten
        pose13[9:12] = pos
        pose13[12] = g
        return pose13
    
    def compute_twist(pose_curr, pose_next):
        """Compute twist from 13D poses using PyPose: ξ = log(T_curr⁻¹ · T_next)."""
        # Extract rotation matrices and positions from 13D poses
        R_curr = pose_curr[:9].reshape(3, 3)
        p_curr = pose_curr[9:12]
        T_curr = np.eye(4, dtype=np.float32)
        T_curr[:3, :3] = R_curr
        T_curr[:3, 3] = p_curr
        
        R_next = pose_next[:9].reshape(3, 3)
        p_next = pose_next[9:12]
        T_next = np.eye(4, dtype=np.float32)
        T_next[:3, :3] = R_next
        T_next[:3, 3] = p_next
        
        # Use PyPose for log map
        T_curr_pp = pp.mat2SE3(torch.tensor(T_curr, dtype=torch.float32).unsqueeze(0))
        T_next_pp = pp.mat2SE3(torch.tensor(T_next, dtype=torch.float32).unsqueeze(0))
        T_rel_pp = T_curr_pp.Inv() @ T_next_pp
        xi = pp.Log(T_rel_pp).tensor().squeeze(0).numpy()
        
        gripper_delta = pose_next[12] - pose_curr[12]
        
        twist7 = np.zeros(7, dtype=np.float32)
        twist7[:6] = xi
        twist7[6] = gripper_delta
        return twist7
    
    def segment_subtasks(props):
        """Segment episode into subtasks based on gripper changes (13D poses)."""
        grippers = props[:, 12]  # Gripper is at index 12 in 13D pose
        is_closed = grippers > 0.5
        changes = np.where(is_closed[1:] != is_closed[:-1])[0] + 1
        boundaries = np.concatenate([[0], changes, [len(props)]])
        
        segments = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1] - 1  # Inclusive end
            if end < start:
                end = start
            
            start_pose = props[start]
            end_pose = props[end]
            
            # Determine task type from gripper change (index 12 in 13D pose)
            if start_pose[12] < 0.5 and end_pose[12] > 0.5:
                task_type = 1  # Pick (gripper closing)
            elif start_pose[12] > 0.5 and end_pose[12] < 0.5:
                task_type = 2  # Place (gripper opening)
            else:
                task_type = 0  # Move (no change)
            
            segments.append({
                'start_idx': start,
                'end_idx': boundaries[i + 1],  # Exclusive for range ops
                'start_pose': start_pose,
                'end_pose': end_pose,
                'task_type': task_type,
            })
        return segments
    
    def get_segment_for_frame(segments, frame_idx):
        """Find which segment a frame belongs to."""
        for seg in segments:
            if seg['start_idx'] <= frame_idx < seg['end_idx']:
                return seg
        return segments[-1] if segments else None
    
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
        MAX_STEPS_PER_EPISODE = 400
        MAX_EPISODES_BEFORE_REFRESH = 100
        PRESTART_AT = 50
        episode_count = 0
        
        for episode in ds:
            if stop_event.is_set():
                break
            
            episode_count += 1
            
            # Extract episode data to numpy
            imgs = []
            props = []
            language_instruction = ""
            
            for s in episode['steps']:
                obs = s['observation']
                
                # Extract language instruction from first step (with fallback)
                if not language_instruction and 'language_instruction' in s:
                    instr = s['language_instruction']
                    if hasattr(instr, 'numpy'):
                        instr = instr.numpy()
                    if isinstance(instr, bytes):
                        instr = instr.decode('utf-8', errors='ignore')
                    language_instruction = str(instr).strip() if instr else ""
                
                # Fallback: check observation/natural_language_instruction
                if not language_instruction and 'natural_language_instruction' in obs:
                    instr = obs['natural_language_instruction']
                    if hasattr(instr, 'numpy'):
                        instr = instr.numpy()
                    if isinstance(instr, bytes):
                        instr = instr.decode('utf-8', errors='ignore')
                    language_instruction = str(instr).strip() if instr else ""
                
                img = None
                for key in img_keys:
                    if key in obs:
                        img = obs[key]
                        break
                
                imgs.append(process_image(img, image_size))
                props.append(get_pose(obs))
            
            # Extract object properties from language instruction
            from imitation.data.goal_oracle import extract_object_properties as extract_props
            object_props = extract_props(language_instruction)
            object_props_vec = np.concatenate([
                object_props['size'],
                object_props['color'],
                object_props['shape']
            ]).astype(np.float32)  # (9,)
            
            if len(imgs) < buffer_size:
                continue
            
            # Truncate very long episodes (keep last N steps)
            if len(imgs) > MAX_STEPS_PER_EPISODE:
                imgs = imgs[-MAX_STEPS_PER_EPISODE:]
                props = props[-MAX_STEPS_PER_EPISODE:]
            
            imgs = np.array(imgs, dtype=np.float32)
            props = np.array(props, dtype=np.float32)
            
            # Segment into subtasks BEFORE generating windows
            segments = segment_subtasks(props)
            
            # Generate windows
            num_windows = len(imgs) - window_size - loss_horizon + 1
            
            for w in range(num_windows):
                if stop_event.is_set():
                    break
                
                w_imgs = imgs[w : w + window_size]
                w_props = props[w : w + window_size]
                
                # Find subtask for current window's last frame
                current_frame = w + window_size - 1
                seg = get_segment_for_frame(segments, current_frame)
                
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
                    # Subtask info for goal computation in main process
                    'subtask_start_pose': seg['start_pose'] if seg else props[0],
                    'subtask_end_pose': seg['end_pose'] if seg else props[-1],
                    'subtask_type': seg['task_type'] if seg else 0,
                    'object_props': object_props_vec,
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
                return
        
        result_queue.put({'error': 'Worker episode loop ended unexpectedly'})
            
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        result_queue.put({'error': error_msg})


class RTXStreamLoader(IterableDataset):
    """
    Unified RTX Dataset Loader with Subtask-Based Goals.
    
    Goals are computed based on subtask boundaries (gripper changes), not window boundaries.
    This means all windows within a pick/place/move segment share the same goal vector.
    
    Args:
        use_subprocess: If True, run TensorFlow in a child process for memory isolation.
                       Recommended for large datasets like DROID.
                       If False, run TF in-process (may leak memory over time).
    """
    
    def __init__(self, dataset_name, split='train', batch_size=1, window_size=8, 
                 loss_horizon=1, image_key=None, image_size=(128, 128), 
                 shuffle_buffer_size=1000, data_dir=None, repeat=True,
                 use_subprocess=True, queue_size=32):
        self.dataset_name = dataset_name
        self.split = split
        self.batch_size = batch_size
        self.window_size = window_size
        self.loss_horizon = loss_horizon
        self.image_key = image_key
        self.image_size = image_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.data_dir = data_dir
        self.repeat = repeat
        self.use_subprocess = use_subprocess
        self.queue_size = queue_size
        
        self.oracle = GoalOracle(output_dim=38)
        
        # Lazy-initialized for in-process mode
        self.builder = None
        self.ds = None
        
    def __len__(self):
        """Returns the number of episodes in the dataset."""
        if self.builder is None:
            self._init_builder()
        if self.builder:
            return self.builder.info.splits[self.split].num_examples
        return 0
    
    def _init_builder(self):
        """Initialize TFDS builder (only for in-process mode or length queries)."""
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        import tensorflow_datasets as tfds
        
        if self.data_dir and self.data_dir.startswith('gs://') and 'fractal' in self.dataset_name:
            full_path = f"{self.data_dir}/{self.dataset_name}/0.1.0"
            self.builder = tfds.builder_from_directory(builder_dir=full_path)
        else:
            use_gcs = (self.data_dir is None)
            self.builder = tfds.builder(self.dataset_name, data_dir=self.data_dir, try_gcs=use_gcs)
    
    def estimate_total_windows(self, avg_episode_length=100):
        """Estimate total windows for progress tracking."""
        num_episodes = len(self)
        if num_episodes == 0:
            return None
        usable_steps = max(0, avg_episode_length - self.window_size - self.loss_horizon + 1)
        return num_episodes * usable_steps
    
    def _start_worker(self, ctx, result_queue, stop_event):
        """Start a new subprocess worker."""
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
    
    def _iter_subprocess(self):
        """Iterator using subprocess for TF memory isolation."""
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue(maxsize=self.queue_size)
        stop_event = ctx.Event()
        
        worker = self._start_worker(ctx, result_queue, stop_event)
        next_worker = None
        next_queue = None
        next_stop_event = None
        
        restart_count = 0
        max_restarts = 100
        consecutive_timeouts = 0
        max_consecutive_timeouts = 90
        
        try:
            while True:
                try:
                    data = result_queue.get(timeout=10.0)
                    consecutive_timeouts = 0
                except queue.Empty:
                    consecutive_timeouts += 1
                    
                    if consecutive_timeouts >= max_consecutive_timeouts:
                        worker.terminate()
                        worker.join(timeout=2.0)
                        if worker.is_alive():
                            worker.kill()
                        consecutive_timeouts = 0
                        restart_count += 1
                        if restart_count > max_restarts:
                            break
                        worker = self._start_worker(ctx, result_queue, stop_event)
                        continue
                    
                    if not worker.is_alive():
                        restart_count += 1
                        if restart_count > max_restarts:
                            break
                        worker = self._start_worker(ctx, result_queue, stop_event)
                    continue
                
                if 'error' in data:
                    restart_count += 1
                    if restart_count > max_restarts:
                        raise RuntimeError(f"Worker error after max restarts: {data['error']}")
                    worker.terminate()
                    worker.join(timeout=2.0)
                    if next_worker is not None:
                        next_stop_event.set()
                        next_worker.terminate()
                        next_worker = None
                    worker = self._start_worker(ctx, result_queue, stop_event)
                    continue
                
                if 'prestart' in data:
                    if next_worker is None:
                        next_queue = ctx.Queue(maxsize=self.queue_size)
                        next_stop_event = ctx.Event()
                        next_worker = self._start_worker(ctx, next_queue, next_stop_event)
                    continue
                
                if 'refresh' in data:
                    worker.join(timeout=2.0)
                    if next_worker is not None:
                        worker = next_worker
                        result_queue = next_queue
                        stop_event = next_stop_event
                        next_worker = None
                        next_queue = None
                        next_stop_event = None
                    else:
                        worker = self._start_worker(ctx, result_queue, stop_event)
                    continue
                
                # Compute goal using subtask poses (from worker)
                # Convert object_props_vec back to dict
                obj_props_vec = data.get('object_props', np.zeros(9, dtype=np.float32))
                object_props = {
                    'size': obj_props_vec[:3],
                    'color': obj_props_vec[3:6],
                    'shape': obj_props_vec[6:9]
                }
                
                goal_emb = self.oracle.encode_goal(
                    data['subtask_type'],
                    data['subtask_start_pose'],
                    data['subtask_end_pose'],
                    object_props
                )
                
                yield {
                    'images': torch.tensor(data['images'], dtype=torch.float32) / 255.0,
                    'proprio': torch.tensor(data['proprio'], dtype=torch.float32),
                    'goal': goal_emb,
                    'actions': torch.tensor(data['actions'], dtype=torch.float32),
                    'target_poses': torch.tensor(data['target_poses'], dtype=torch.float32),
                }
                
        finally:
            stop_event.set()
            worker.terminate()
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.kill()
    
    def _quat_to_rotmat(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix."""
        quat = quat / (np.linalg.norm(quat) + 1e-8)
        return R.from_quat(quat).as_matrix().astype(np.float32)
    
    def _get_pose(self, obs):
        """Extract 13D SE(3) pose: [R_flat(9), pos(3), gripper(1)]."""
        pos = np.zeros(3, dtype=np.float32)
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        rotmat = None
        
        if 'base_pose_tool_reached' in obs:
            p7 = obs['base_pose_tool_reached'].numpy()
            pos = p7[:3]
            quat = p7[3:7]
        elif 'cartesian_position' in obs:
            p6 = obs['cartesian_position'].numpy()
            if p6.shape[0] == 6:
                pos = p6[:3]
                euler = p6[3:]
                rotmat = R.from_euler('xyz', euler).as_matrix().astype(np.float32)
        elif 'ee_pose' in obs:
            p7 = obs['ee_pose'].numpy()
            pos = p7[:3]
            if p7.shape[0] >= 7:
                quat = p7[3:7]
        elif 'pose' in obs:
            p7 = obs['pose'].numpy()
            pos = p7[:3]
            if p7.shape[0] >= 7:
                quat = p7[3:7]
        
        if rotmat is None:
            rotmat = self._quat_to_rotmat(quat)
        
        g = 0.0
        if 'gripper_closed' in obs:
            g = obs['gripper_closed'].numpy()
        elif 'gripper_position' in obs:
            g = obs['gripper_position'].numpy()
        elif 'gripper_state' in obs:
            g = obs['gripper_state'].numpy()
        
        if not np.isscalar(g):
            g = g.item() if g.size == 1 else g[0]
        
        # 13D pose: [R_flat(9), pos(3), gripper(1)]
        pose13 = np.zeros(13, dtype=np.float32)
        pose13[:9] = rotmat.flatten()
        pose13[9:12] = pos
        pose13[12] = g
        return pose13
    
    def _compute_twist(self, pose_curr: np.ndarray, pose_next: np.ndarray) -> np.ndarray:
        """Compute twist from 13D poses using PyPose: ξ = log(T_curr⁻¹ · T_next)."""
        R_curr = pose_curr[:9].reshape(3, 3)
        p_curr = pose_curr[9:12]
        T_curr = np.eye(4, dtype=np.float32)
        T_curr[:3, :3] = R_curr
        T_curr[:3, 3] = p_curr
        
        R_next = pose_next[:9].reshape(3, 3)
        p_next = pose_next[9:12]
        T_next = np.eye(4, dtype=np.float32)
        T_next[:3, :3] = R_next
        T_next[:3, 3] = p_next
        
        # Use PyPose for log map
        T_curr_pp = pp.mat2SE3(torch.tensor(T_curr, dtype=torch.float32).unsqueeze(0))
        T_next_pp = pp.mat2SE3(torch.tensor(T_next, dtype=torch.float32).unsqueeze(0))
        T_rel_pp = T_curr_pp.Inv() @ T_next_pp
        xi = pp.Log(T_rel_pp).tensor().squeeze(0).numpy()
        
        gripper_delta = pose_next[12] - pose_curr[12]
        
        twist7 = np.zeros(7, dtype=np.float32)
        twist7[:6] = xi
        twist7[6] = gripper_delta
        return twist7
    
    def _process_image(self, img):
        """Process image using cv2."""
        import cv2
        
        if img is None:
            return np.zeros((3, 128, 128), dtype=np.float32)
        
        if hasattr(img, 'numpy'):
            img = img.numpy()
        
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return img
    
    def _segment_subtasks(self, props: np.ndarray, object_props: dict = None):
        """Segment episode into subtasks based on gripper changes (13D poses)."""
        grippers = props[:, 12]  # Gripper at index 12 in 13D pose
        is_closed = grippers > 0.5
        changes = np.where(is_closed[1:] != is_closed[:-1])[0] + 1
        boundaries = np.concatenate([[0], changes, [len(props)]])
        
        segments = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1] - 1
            if end < start:
                end = start
            
            start_pose = props[start]
            end_pose = props[end]
            
            if start_pose[12] < 0.5 and end_pose[12] > 0.5:
                task_type = 1  # Pick
            elif start_pose[12] > 0.5 and end_pose[12] < 0.5:
                task_type = 2  # Place
            else:
                task_type = 0  # Move
            
            goal = self.oracle.encode_goal(task_type, start_pose, end_pose, object_props)
            segments.append({
                'start': start,
                'end': boundaries[i + 1],
                'goal': goal
            })
        return segments
    
    def _get_goal_for_frame(self, segments, frame_idx):
        """Get goal for a given frame index."""
        for seg in segments:
            if seg['start'] <= frame_idx < seg['end']:
                return seg['goal']
        return segments[-1]['goal'] if segments else None
    
    def _iter_inprocess(self):
        """Iterator using in-process TensorFlow (may leak memory)."""
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        tf.config.run_functions_eagerly(True)
        import tensorflow_datasets as tfds
        import cv2
        
        if self.builder is None:
            self._init_builder()
        
        read_config = tfds.ReadConfig(try_autocache=False, add_tfds_id=False)
        
        if self.ds is None:
            self.ds = self.builder.as_dataset(split=self.split, shuffle_files=True, read_config=read_config)
        
        ds = self.ds
        
        options = tf.data.Options()
        options.experimental_optimization.apply_default_optimizations = False
        options.autotune.enabled = False
        ds = ds.with_options(options)
        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            ds = ds.shard(num_shards=worker_info.num_workers, index=worker_info.id)
        
        if self.repeat:
            ds = ds.repeat()
        
        if self.shuffle_buffer_size > 1:
            ds = ds.shuffle(self.shuffle_buffer_size)
        
        fallback_keys = ['image', 'exterior_image_1_left', 'wrist_image_left', 'exterior_image_2_left']
        image_keys = [self.image_key] if self.image_key else fallback_keys
        buffer_size = self.window_size + self.loss_horizon
        
        for episode in ds:
            # Extract full episode
            imgs = []
            props = []
            language_instruction = ""
            
            for s in episode['steps']:
                obs = s['observation']
                
                # Extract language instruction from first step (with fallback)
                if not language_instruction and 'language_instruction' in s:
                    instr = s['language_instruction']
                    if hasattr(instr, 'numpy'):
                        instr = instr.numpy()
                    if isinstance(instr, bytes):
                        instr = instr.decode('utf-8', errors='ignore')
                    language_instruction = str(instr).strip() if instr else ""
                
                # Fallback: check observation/natural_language_instruction
                if not language_instruction and 'natural_language_instruction' in obs:
                    instr = obs['natural_language_instruction']
                    if hasattr(instr, 'numpy'):
                        instr = instr.numpy()
                    if isinstance(instr, bytes):
                        instr = instr.decode('utf-8', errors='ignore')
                    language_instruction = str(instr).strip() if instr else ""
                
                img = None
                for key in image_keys:
                    if key in obs:
                        img = obs[key]
                        break
                
                imgs.append(self._process_image(img))
                props.append(self._get_pose(obs))
            
            if len(imgs) < buffer_size:
                continue
            
            # Truncate very long episodes (keep last N steps) - same as subprocess mode
            MAX_STEPS_PER_EPISODE = 400
            if len(imgs) > MAX_STEPS_PER_EPISODE:
                imgs = imgs[-MAX_STEPS_PER_EPISODE:]
                props = props[-MAX_STEPS_PER_EPISODE:]
            
            imgs = np.array(imgs, dtype=np.float32)
            props = np.array(props, dtype=np.float32)
            
            # Extract object properties from language instruction
            object_props = extract_object_properties(language_instruction)
            
            # Segment into subtasks (with object_props)
            segments = self._segment_subtasks(props, object_props)
            
            # Generate windows
            num_windows = len(imgs) - self.window_size - self.loss_horizon + 1
            
            for w in range(num_windows):
                w_imgs = imgs[w : w + self.window_size]
                w_props = props[w : w + self.window_size]
                
                current_frame = w + self.window_size - 1
                goal_emb = self._get_goal_for_frame(segments, current_frame)
                
                if goal_emb is None:
                    continue
                
                # Compute twists
                target_twists = []
                target_poses = []
                start_idx = w + self.window_size - 1
                
                for h in range(self.loss_horizon):
                    curr_idx = start_idx + h
                    next_idx = curr_idx + 1
                    twist = self._compute_twist(props[curr_idx], props[next_idx])
                    target_twists.append(twist)
                    target_poses.append(props[next_idx])
                
                yield {
                    'images': torch.tensor(w_imgs, dtype=torch.float32),
                    'proprio': torch.tensor(w_props, dtype=torch.float32),
                    'goal': goal_emb,
                    'actions': torch.tensor(np.array(target_twists), dtype=torch.float32),
                    'target_poses': torch.tensor(np.array(target_poses), dtype=torch.float32),
                }
            
            # Cleanup
            tf.keras.backend.clear_session()
            gc.collect()
    
    def __iter__(self):
        if self.use_subprocess:
            yield from self._iter_subprocess()
        else:
            yield from self._iter_inprocess()
