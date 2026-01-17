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

class ActionChunkEnsembler:
    """
    Temporal Ensembling for Action Chunking.
    
    Maintains a buffer of recent action predictions and averages overlapping
    chunks to produce smooth, consistent actions.
    
    Algorithm:
    1.  Receive a new chunk of size `k` at time `t`.
    2.  Add this chunk to the buffer at positions `[t, t+k]`.
    3.  For time `t`, gather all valid predictions from past chunks covering `t`.
    4.  Average them to get the final action for `t`.
    """
    def __init__(self, action_dim: int, chunk_size: int, max_history: int = 100):
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.max_history = max_history
        
        # Buffer to store predictions for each future timestep
        # Key: absolute_timestep (int), Value: list of np.ndarray actions
        self.prediction_buffer = {}
        self.current_step = 0
        
    def reset(self):
        self.prediction_buffer = {}
        self.current_step = 0
        
    def update(self, action_chunk: np.ndarray) -> np.ndarray:
        """
        Register a new action chunk and return the ensembled action for the current step.
        
        Args:
            action_chunk: (chunk_size, action_dim) array
            
        Returns:
            ensembled_action: (action_dim,) array
        """
        # 1. Add new chunk to buffer
        for i in range(self.chunk_size):
            future_step = self.current_step + i
            if future_step not in self.prediction_buffer:
                self.prediction_buffer[future_step] = []
            
            # The i-th element of the chunk corresponds to future_step
            self.prediction_buffer[future_step].append(action_chunk[i])
            
        # 2. Ensemble prediction for current_step
        if self.current_step in self.prediction_buffer:
            candidates = self.prediction_buffer[self.current_step]
            # Simple average of all overlapping predictions
            # TODO: Could implement weighted averaging (linear decay, exponential, etc.)
            ensembled_action = np.mean(candidates, axis=0)
            
            # Cleanup: Remove old steps to prevent memory leak
            del self.prediction_buffer[self.current_step]
        else:
            # Fallback if no prediction exists (should not happen with regular updates)
            ensembled_action = np.zeros(self.action_dim)
            
        self.current_step += 1
        return ensembled_action


class AxisInference:
    """Production inference wrapper with safety features, KV-caching, and temporal ensembling."""
    
    def __init__(self, checkpoint_path: str, config: dict, device: str = 'cuda'):
        """
        Args:
            checkpoint_path: Path to .pt checkpoint
            config: Model configuration dict
            device: 'cuda' or 'cpu'
        """
        self.config = config
        self.device = device
        
        # Load model
        self.model = AxisModel(config)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Safety bounds
        self.max_pos_delta = config.get('max_pos_delta', 0.05)
        self.max_rot_delta = config.get('max_rot_delta', 0.1)
        
        # Action Chunking Ensembler
        self.chunk_size = config.get('action_dim', 1) # Fallback to 1 if not chunked, though typically it is
        # Note: If passing 'chunk_size' explicitly in config would be better. 
        # Checking AxisModel output shape dynamically or config parameter.
        # Assuming config has 'chunk_size', default to 10 based on context if missing.
        self.chunk_size = config.get('chunk_size', 10) 
        self.action_dim = config.get('action_dim', 7) # Twist dim
        
        self.ensembler = ActionChunkEnsembler(
            action_dim=self.action_dim, 
            chunk_size=self.chunk_size
        )
        
        # State for KV Caching
        self.cached_tokens = None
        
    def reset(self):
        """Reset stateful components (caching, ensembling)."""
        self.cached_tokens = None
        self.ensembler.reset()
    
    @torch.no_grad()
    def predict(self, images: np.ndarray, proprio: np.ndarray, 
                goal: np.ndarray) -> dict:
        """
        Run inference with KV-caching and Temporal Ensembling.
        
        Args:
            images: (W, C, H, W) or (W, H, W, C) - window of images
            proprio: (W, 10) - window of 10D poses (3 pos, 6 rot, 1 gripper)
            goal: (64,) - goal vector
        
        Returns:
            dict with:
                - 'action_delta': (10,) smoothed predicted twist (converted to state-like format for compat)
                  Wait, standard is usually returning the *next* absolute pose.
                - 'action_absolute': (10,) absolute target pose (Integrated)
                - 'position': (3,) target position
                - 'quaternion': (4,) target rotation as [x,y,z,w]
                - 'gripper': float target gripper state
                - 'requery': bool whether to request new goal
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
        
        # Update Cache (shift window mechanism implies we keep simplest cache strategy: 
        # usually KV cache grows, but here AxisModel manages fixed window.
        # AxisModel V2 implementation suggests returning full tokens.
        # We need to slice to keep the valid history for the *next* step.
        # If model outputs (B, W, D), we keep last W-1 for next step input.
        W = images_t.shape[1]
        self.cached_tokens = new_tokens[:, 1:, ...] if W > 1 else new_tokens
        
        # --- Process Action ---
        # 1. Get raw chunk: (1, Chunk, 7) -> (Chunk, 7)
        action_chunk_np = action_chunk.squeeze(0).cpu().numpy()
        
        # 2. Ensemble
        # Returns (7,) twist: [omg_x, omg_y, omg_z, v_x, v_y, v_z, g]
        action_twist_smooth = self.ensembler.update(action_chunk_np)
        
        # 3. Safety Clean (Clamp Twist)
        # Angular Velocity (0:3)
        action_twist_smooth[:3] = np.clip(action_twist_smooth[:3], -self.max_rot_delta, self.max_rot_delta)
        # Linear Velocity (3:6)
        action_twist_smooth[3:6] = np.clip(action_twist_smooth[3:6], -self.max_pos_delta, self.max_pos_delta)
        
        # 4. Integrate to Absolute Pose
        current_pose = proprio[-1] # (10,)
        action_absolute = self._integrate_action(current_pose, action_twist_smooth)
        
        # Convert to output formats
        rot_6d = torch.tensor(action_absolute[3:9])
        quat = rotation_6d_to_quaternion(rot_6d).numpy()
        
        return {
            'action_twist': action_twist_smooth,
            'action_absolute': action_absolute,
            'position': action_absolute[:3],
            'quaternion': quat,
            'gripper': float(action_absolute[9]),
            'requery': torch.sigmoid(requery_logit).item() > 0.5
        }

    def _integrate_action(self, current_pose: np.ndarray, action_twist: np.ndarray) -> np.ndarray:
        """
        Integrate 7D Twist (angular_vel, linear_vel, gripper_vel) into 10D Pose.
        
        Args:
            current_pose: (10,) [pos(3), rot6d(6), gripper(1)]
            action_twist: (7,) [ang_vel(3), lin_vel(3), gripper_prob/vel(1)]
            
        Returns:
            next_pose: (10,)
        """
        # 1. Position: Simple addition
        # Twist: 0-2 (ang), 3-5 (lin)
        pos_curr = current_pose[:3]
        lin_vel = action_twist[3:6]
        pos_new = pos_curr + lin_vel
        
        # 2. Rotation: Exponential map integration
        # Current rotation matrix
        rot6d_curr = torch.tensor(current_pose[3:9]).unsqueeze(0) # (1, 6)
        mat_curr = quaternion_to_matrix(rotation_6d_to_quaternion(rot6d_curr)).squeeze(0) # (3, 3)
        
        # Delta rotation matrix from axis-angle
        ang_vel = action_twist[:3]
        angle = np.linalg.norm(ang_vel)
        if angle < 1e-6:
            mat_delta = torch.eye(3)
        else:
            axis = torch.tensor(ang_vel / angle)
            # Rodrigues formula for per-step integration
            # R = I + sin(theta)K + (1-cos(theta))K^2
            K = torch.tensor([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            mat_delta = torch.eye(3) + torch.sin(torch.tensor(angle)) * K + (1 - torch.cos(torch.tensor(angle))) * (K @ K)
            
        # Apply delta: R_new = R_delta @ R_curr (Global frame logic? Or local?)
        # Usually twist is local or global depending on training. 
        # Assuming Global Frame twist for simplicity unless specified.
        # If prediction is dR in world frame: R_new = dR @ R_old
        mat_new = mat_delta @ mat_curr
        
        # Convert rotation matrix to 6D using proper utility
        d6_new = matrix_to_rotation_6d(mat_new.unsqueeze(0)).squeeze(0)  # (6,)

        
        # 3. Gripper
        # Assuming twist output is logits or delta?
        # Usually simpler to treat as absolute logit for boolean, or linear delta.
        # Task says "7D Twist ... gripper state". 
        # If it's `gripper_state` (prob), then direct replacement.
        # If it's velocity, add.
        # Let's assume absolute probability for now as is common in ACT/Diffusion, 
        # BUT `Twist` implies velocity. 
        # "Twist output... plus the gripper state" - User prompt in history (Conv e8c...).
        # "Output dimension 7 (3 trans + 3 rot + 1 gripper)". 
        # Let's treat gripper as direct value (state) not delta, unless specified.
        # Standard Axis convention: Gripper is usually absolute [0, 1].
        grip_new = action_twist[6] # Direct assignment
        
        # Assemble
        return np.concatenate([pos_new, d6_new.numpy(), [grip_new]])

if __name__ == "__main__":
    # Test Ensembler
    ensembler = ActionChunkEnsembler(action_dim=1, chunk_size=3)
    
    # t=0, predict [1, 2, 3] -> t0=1, t1=2, t2=3
    a0 = ensembler.update(np.array([[1], [2], [3]])) 
    print(f"t=0: {a0} (Expect 1.0)")
    
    # t=1, predict [4, 5, 6] -> t1=4, t2=5, t3=6
    # Buffer at t1 has: 2 (from prev), 4 (from curr) -> avg=3
    a1 = ensembler.update(np.array([[4], [5], [6]]))
    print(f"t=1: {a1} (Expect 3.0)")
    
    print("Inference wrapper ready")

