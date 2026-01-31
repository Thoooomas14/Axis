"""
Production inference wrapper for Axis V2.

Provides a clean API for running inference with safety features,
rotation conversion, action smoothing (via Temporal Ensembling), 
and state management for deployment.
"""

import torch
import numpy as np
from collections import deque
from src.models.axis import AxisModel
from src.utils.rotation_utils import rotation_6d_to_quaternion, quaternion_to_matrix, matrix_to_quaternion, matrix_to_rotation_6d
import torch.nn.functional as F

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
            return torch.zeros(chunk.shape[-1], device=chunk.device) if chunk is not None else torch.zeros(7)
            
        # Weighted Average
        actions_stack = torch.stack(actions, dim=0) # (N, 7)
        weights_stack = torch.tensor(weights, device=actions_stack.device).unsqueeze(1) # (N, 1)
        
        weighted_sum = (actions_stack * weights_stack).sum(dim=0)
        total_weight = weights_stack.sum()
        
        return weighted_sum / total_weight


class AxisInference:
    """Production inference wrapper with safety features, KV-caching, and temporal ensembling."""
    
    def __init__(self, checkpoint_path: str, config: dict, device: str = 'cuda', random_weights=False, ensemble_k=0.01):
        """
        Args:
            checkpoint_path: Path to .pt checkpoint
            config: Model configuration dict
            device: 'cuda' or 'cpu'
            ensemble_k: Exponential weighting factor for ensembling
        """
        self.device = device
        self.config = config
        self.random_weights = random_weights
        self.ensemble_k = ensemble_k
        
        # Initialize Model
        self.model = AxisModel(config).to(device)
        self.model.eval()
        
        if self.random_weights:
            print("WARNING: Using Random Weights for AxisInference.")
        else:
            self._load_checkpoint(checkpoint_path)
            
        # State History (for windowing)
        self.image_history = deque(maxlen=config.get('window_size', 1))
        self.proprio_history = deque(maxlen=config.get('window_size', 1))
        
        self.cached_tokens = None
        
        # Action Chunking Ensembler
        self.chunk_size = config.get('chunk_size', 10) 
        self.action_dim = config.get('action_dim', 7) # Twist dim
        
        self.ensembler = TemporalEnsembler(
            horizon=self.chunk_size,
            k=self.ensemble_k
        )
        
        self.current_step = 0
        
    def _load_checkpoint(self, path):
        if not path: return
        print(f"Loading checkpoint {path}...")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)

    def reset(self):
        """Reset stateful components (caching, ensembling)."""
        self.cached_tokens = None
        self.ensembler.reset()
        self.current_step = 0
    
    @torch.no_grad()
    def predict(self, images: np.ndarray, proprio: np.ndarray, 
                goal: np.ndarray) -> dict:
        """
        Run inference with KV-caching and Temporal Ensembling.
        
        Args:
            images: (W, C, H, W) - window of images
            proprio: (W, 13) - window of 13D poses [R_flat(9), pos(3), grip(1)] in MILLIMETERS
            goal: (38,) - goal vector
        """
        # Convert to tensors and add batch dim
        images_t = torch.tensor(images, dtype=torch.float32).unsqueeze(0).to(self.device)
        proprio_t = torch.tensor(proprio, dtype=torch.float32).unsqueeze(0).to(self.device)
        goal_t = torch.tensor(goal, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Forward pass with Caching
        # Returns: pred_action (B, Chunk, 7), requery (B, 1), tokens (B, W, D)
        action_chunk, requery_logit, new_tokens = self.model(
            images_t, proprio_t, goal_t, 
            cached_tokens=self.cached_tokens,
            return_tokens=True
        )
        
        # Update Cache
        W = images_t.shape[1]
        self.cached_tokens = new_tokens[:, 1:, ...] if W > 1 else new_tokens
        
        # --- Process Action ---
        # 1. Get raw chunk: (1, Chunk, 7) -> (Chunk, 7)
        action_chunk_prev = action_chunk.squeeze(0).cpu() # Keep as tensor for ensembler
        
        # 2. Ensemble
        self.ensembler.update(self.current_step, action_chunk_prev)
        action_twist_smooth = self.ensembler.get_action(self.current_step).numpy() # (7,)
        
        # Increment internal step counter
        self.current_step += 1
        
        # 3. Safety Clean (Clamp Twist)
        # Linear Velocity (0:3) - PyPose Convention
        action_twist_smooth[:3] = np.clip(action_twist_smooth[:3], -self.config.get('max_pos_delta', 10.0), self.config.get('max_pos_delta', 10.0))
        # Angular Velocity (3:6)
        action_twist_smooth[3:6] = np.clip(action_twist_smooth[3:6], -self.config.get('max_rot_delta', 0.1), self.config.get('max_rot_delta', 0.1))
        
        # 4. Integrate to Absolute Pose
        # Current pose is the last one in the window
        current_pose = proprio[-1] # (13,)
        action_absolute = self._integrate_action_13d(current_pose, action_twist_smooth)
        
        # Output is 13D pose in mm
        # Extract Quaternion for Isaac Sim compat if needed
        # But Isaac script will likely need to handle conversion.
        # We return the 13D absolute pose.
        
        # Recover quaternion for convenience
        rot_mat = action_absolute[:9].reshape(3, 3)
        quat = matrix_to_quaternion(torch.tensor(rot_mat).unsqueeze(0)).squeeze(0).numpy() # (4,) [x,y,z,w]
        
        return {
            'action_twist': action_twist_smooth,
            'action_absolute': action_absolute, # 13D
            'position': action_absolute[9:12], # mm
            'quaternion': quat, # [x,y,z,w]
            'gripper': float(action_absolute[12]),
            'requery': torch.sigmoid(requery_logit).item() > 0.5
        }

    def _integrate_action_13d(self, current_pose: np.ndarray, action_twist: np.ndarray) -> np.ndarray:
        """
        Integrate 7D Twist into 13D Pose (Millimeters).
        
        Args:
            current_pose: (13,) [R_flat(9), pos(3), gripper(1)]
            action_twist: (7,) [ang_vel(3), lin_vel(3), gripper_prob(1)]
            
        Returns:
            next_pose: (13,)
        """
        # 2. Rotation: Exponential map integration
        # Current rotation matrix
        mat_curr = torch.tensor(current_pose[:9].reshape(3, 3), dtype=torch.float32)

        # 1. Position: Body Frame Twist -> World Frame Update
        # T_next = T_curr * Exp(xi)
        # p_next = p_curr + R_curr * v_body
        # PyPose Convention: [Linear(3), Angular(3)]
        pos_curr = torch.tensor(current_pose[9:12], dtype=torch.float32)
        lin_vel_mm = torch.tensor(action_twist[:3], dtype=torch.float32) # Linear is first 3
        
        # Rotate linear velocity to world frame
        lin_vel_world = mat_curr @ lin_vel_mm
        pos_new = pos_curr + lin_vel_world
        
        pos_new = pos_new.numpy()
        
        # Delta rotation matrix from axis-angle
        ang_vel = action_twist[3:6] # Angular is next 3
        angle = np.linalg.norm(ang_vel)
        if angle < 1e-6:
            mat_delta = torch.eye(3, dtype=torch.float32)
        else:
            axis = torch.tensor(ang_vel / angle, dtype=torch.float32)
            K = torch.tensor([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ], dtype=torch.float32)
            angle_t = torch.tensor(angle, dtype=torch.float32)
            mat_delta = torch.eye(3, dtype=torch.float32) + torch.sin(angle_t) * K + (1 - torch.cos(angle_t)) * (K @ K)
            
        # Apply delta: R_new = R_curr @ R_delta (Body Frame Twist)
        # Training used: T_curr.Inv() @ T_next -> Delta is in Body Frame
        # So integration must be R_new = R_curr @ R_delta
        mat_new = mat_curr @ mat_delta
        
        # 3. Gripper
        # Model output is delta (from rtx_stream_loader logic)
        grip_curr = current_pose[12]
        grip_delta = action_twist[6]
        grip_new = np.clip(grip_curr + grip_delta, 0.0, 1.0)
        
        # Re-orthonormalize rotation matrix using SVD to prevent drift
        # R = U @ V.T
        u, s, v = torch.svd(mat_new)
        # Check determinant to prevent reflection
        rot_maybe = u @ v.T
        if torch.det(rot_maybe) < 0:
            u[:, -1] *= -1
            mat_new = u @ v.T
        else:
            mat_new = rot_maybe
        
        # Assemble 13D
        pose_new = np.zeros(13, dtype=np.float32)
        pose_new[:9] = mat_new.flatten().numpy()
        pose_new[9:12] = pos_new
        pose_new[12] = grip_new
        
        return pose_new

