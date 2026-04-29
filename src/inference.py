"""
Production inference wrapper for Axis V2.

Provides a clean API for running inference with safety features,
rotation conversion, action smoothing (via Temporal Ensembling),
and state management for deployment.
"""

import logging

import torch
import numpy as np
from collections import deque
from src.models.axis import AxisModel
from src.utils.rotation_utils import matrix_to_quaternion
import pypose as pp


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
            # Fallback if no coverage
            return torch.zeros(7)

        # Weighted Average
        actions_stack = torch.stack(actions, dim=0)  # (N, 7)
        weights_stack = torch.tensor(weights, device=actions_stack.device).unsqueeze(
            1
        )  # (N, 1)

        weighted_sum = (actions_stack * weights_stack).sum(dim=0)
        total_weight = weights_stack.sum()

        return weighted_sum / total_weight


class AxisInference:
    """Production inference wrapper with safety features, KV-caching, and temporal ensembling."""

    def __init__(
        self,
        checkpoint_path: str,
        config: dict | str,
        device: str = "cuda",
        random_weights=False,
        ensemble_k=0.01,
    ):
        """
        Args:
            checkpoint_path: Path to .pt checkpoint
            config: Model configuration dict
            device: 'cuda' or 'cpu'
            ensemble_k: Exponential weighting factor for ensembling
        """
        self.device = device
        self.random_weights = random_weights
        self.ensemble_k = ensemble_k

        # Initialize Model
        self.model = AxisModel(config).to(device)
        config = self.model.config  # Get config from model (handles defaults)
        self.config = config
        self.model.eval()
        self.window_size = self.config.get("window_size", 10)

        if self.random_weights:
            logging.warning("WARNING: Using Random Weights for AxisInference.")
        else:
            self._load_checkpoint(checkpoint_path)

        # State History (for windowing)
        self.image_history = deque(maxlen=config.get("window_size", 1))
        self.proprio_history = deque(maxlen=config.get("window_size", 1))

        self.cached_tokens = None

        # Action Chunking Ensembler
        self.chunk_size = config.get("chunk_size", 10)
        self.action_dim = config.get("action_dim", 7)  # Twist dim

        self.ensembler = TemporalEnsembler(horizon=self.chunk_size, k=self.ensemble_k)

        self.current_step = 0

    def _load_checkpoint(self, path):
        if not path:
            return
        logging.info(f"Loading checkpoint {path}...")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)

    def reset(self):
        """Reset stateful components (caching, ensembling)."""
        self.cached_tokens = None
        self.ensembler.reset()
        self.current_step = 0

    @torch.no_grad()
    def predict(
        self, images: np.ndarray, proprio: np.ndarray, goal: np.ndarray
    ) -> dict:
        """
        Run inference with KV-caching and Temporal Ensembling.

        Args:
            images: (W, C, H, W) - window of images
            proprio: (W, 13) - window of 13D poses [R_flat(9), pos(3), grip(1)] in MILLIMETERS
            goal: (38,) - goal vector
        """
        # Convert to tensors and add batch dim
        images_t = (
            torch.tensor(images, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        proprio_t = (
            torch.tensor(proprio, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        goal_t = torch.tensor(goal, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Forward pass with Caching
        # Returns: pred_action (B, Chunk, 7), requery (B, 1), tokens (B, W, D)
        action_chunk, requery_logit, new_tokens = self.model(
            images_t,
            proprio_t,
            goal_t,
            cached_tokens=self.cached_tokens,
            return_tokens=True,
        )

        # Update Cache
        W = images_t.shape[1]
        self.cached_tokens = new_tokens[:, 1:, ...] if W > 1 else new_tokens

        # --- Process Action ---
        # 1. Get raw chunk: (1, Chunk, 7) -> (Chunk, 7)
        action_chunk_prev = action_chunk.squeeze(
            0
        ).cpu()  # Keep as tensor for ensembler

        # 2. Ensemble
        self.ensembler.update(self.current_step, action_chunk_prev)
        action_twist_smooth = self.ensembler.get_action(
            self.current_step
        ).numpy()  # (7,)

        # Increment internal step counter
        self.current_step += 1

        # 3. Safety Clamp (very permissive — make_gif uses NO clamp)
        # PyPose se3 Convention: [v(3), ω(3), grip(1)]
        # NOTE: PyPose uses [Translation, Rotation]!
        max_rot = self.config.get("max_rot_delta", 1.0)
        max_pos = self.config.get("max_pos_delta", 50.0)

        # Linear Velocity (0:3)
        action_twist_smooth[:3] = np.clip(action_twist_smooth[:3], -max_pos, max_pos)

        # Angular Velocity (3:6)
        ang_before = action_twist_smooth[3:6].copy()
        action_twist_smooth[3:6] = np.clip(action_twist_smooth[3:6], -max_rot, max_rot)
        if np.any(np.abs(ang_before) > max_rot):
            logging.debug(
                f"Angular clipped: {ang_before.round(3)} → {action_twist_smooth[3:6].round(3)}"
            )

        # 4. Integrate to Absolute Pose
        # Current pose is the last one in the window
        current_pose = proprio[-1]  # (13,)
        action_absolute = self._integrate_action_13d(current_pose, action_twist_smooth)

        # Output is 13D pose in mm
        # Extract Quaternion for Isaac Sim compat if needed
        # But Isaac script will likely need to handle conversion.
        # We return the 13D absolute pose.

        # Recover quaternion for convenience
        rot_mat = action_absolute[:9].reshape(3, 3)
        quat = (
            matrix_to_quaternion(torch.tensor(rot_mat).unsqueeze(0)).squeeze(0).numpy()
        )  # (4,) [x,y,z,w]

        return {
            "action_twist": action_twist_smooth,
            "action_absolute": action_absolute,  # 13D
            "position": action_absolute[9:12],  # mm
            "quaternion": quat,  # [x,y,z,w]
            "gripper": float(action_absolute[12]),
            "requery": torch.sigmoid(requery_logit).item() > 0.5,
            "requery_prob": torch.sigmoid(requery_logit).item(),
        }

    def _integrate_action_13d(
        self, current_pose: np.ndarray, action_twist: np.ndarray
    ) -> np.ndarray:
        """
        Integrate 7D Twist into 13D Pose (Millimeters) using PyPose.
        Matches logic in imitation/train.py: endpoint_chordal_loss.

        Args:
            current_pose: (13,) [R_flat(9), pos(3), gripper(1)]
            action_twist: (7,) [ang_vel(3), lin_vel(3), gripper_delta(1)]

        Returns:
            next_pose: (13,)
        """
        device = self.device
        dtype = torch.float32

        # Prepare inputs: (1, 13) and (1, 7)
        current_pose_t = torch.tensor(
            current_pose, device=device, dtype=dtype
        ).unsqueeze(0)
        action_twist_t = torch.tensor(
            action_twist, device=device, dtype=dtype
        ).unsqueeze(0)

        # 1. Build Current SE(3) Transform
        R_curr = current_pose_t[:, :9].reshape(1, 3, 3)
        p_curr = current_pose_t[:, 9:12]

        T_curr = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)

        # Sanitize input rotation (Orthonormalize)
        # Matches orthonormalize_rotation in train.py
        U, S, V = torch.svd(R_curr)
        with torch.no_grad():
            det = torch.det(U @ V.transpose(-2, -1))
            diag = torch.ones_like(S)
            diag[:, -1] = det
            R_curr_clean = U @ torch.diag_embed(diag) @ V.transpose(-2, -1)

        T_curr[:, :3, :3] = R_curr_clean
        T_curr[:, :3, 3] = p_curr

        # Initialize LieTensor
        T_curr_pp = pp.from_matrix(T_curr, ltype=pp.SE3_type, check=False)

        # 2. Apply Twist (Exponential Map on SE(3))
        twist_6d = action_twist_t[:, :6]
        gripper_delta = action_twist_t[:, 6:7]

        T_delta_pp = pp.Exp(pp.se3(twist_6d))
        T_next_pp = T_curr_pp @ T_delta_pp

        # 3. Handle Gripper
        grip_curr = current_pose_t[:, 12:13]
        grip_next = torch.clamp(grip_curr + gripper_delta, 0.0, 1.0)

        # 4. Extract and Assemble Result
        T_next = T_next_pp.matrix()  # (1, 4, 4)
        R_next = T_next[:, :3, :3]
        p_next = T_next[:, :3, 3]

        pose_new = np.zeros(13, dtype=np.float32)
        pose_new[:9] = R_next.flatten().cpu().numpy()
        pose_new[9:12] = p_next.flatten().cpu().numpy()
        pose_new[12] = grip_next.item()

        return pose_new
