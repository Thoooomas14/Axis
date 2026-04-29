import torch
import gymnasium as gym
import pypose as pp

class AxisObservationWrapper(gym.Wrapper):
    """
    Wraps the Isaac Lab environment to provide history-windowed observations
    compatible with the Axis Transformer Model.

    Returns:
        images: (B, Window, C, H, W)
        proprio: (B, Window, 13) -> [Rot9(9), Pos_mm(3), Gripper(1)]
    """

    def __init__(self, env, window_size=10, device="cpu"):
        super().__init__(env)
        self.window_size = window_size
        self.device = device
        self.image_buffer = None
        self.proprio_buffer = None
        self.num_envs = getattr(env, "num_envs", 1)
        self.proprio_dim = 13  # Axis V2 uses 13D

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if "policy" not in obs:
            pass
        img_current = self._process_image(obs)
        prop_current = self._process_proprio(obs)
        B = img_current.shape[0]
        C, H, W = img_current.shape[1], img_current.shape[2], img_current.shape[3]

        # Initialize buffers
        self.image_buffer = torch.zeros(
            B, self.window_size, C, H, W, device=self.device
        )
        self.proprio_buffer = torch.zeros(
            B, self.window_size, self.proprio_dim, device=self.device
        )

        # Fill with current observation (repeat for initial window)
        self.image_buffer[:] = img_current.unsqueeze(1)
        self.proprio_buffer[:] = prop_current.unsqueeze(1)

        return self._get_obs(), info

    def step(self, action, model_run=True):
        obs, rew, terminated, truncated, info = self.env.step(action)
        if model_run:
            img_current = self._process_image(obs)
            prop_current = self._process_proprio(obs)

            # Shift buffer and append new
            self.image_buffer = torch.roll(self.image_buffer, shifts=-1, dims=1)
            self.image_buffer[:, -1] = img_current

            self.proprio_buffer = torch.roll(self.proprio_buffer, shifts=-1, dims=1)
            self.proprio_buffer[:, -1] = prop_current

        return self._get_obs(), rew, terminated, truncated, info

    def _get_obs(self):
        return {
            "images": self.image_buffer.clone(),
            "proprio": self.proprio_buffer.clone(),
        }

    def _process_image(self, obs):
        rgb = obs.get("policy", {}).get("rgb", None)
        if rgb is None:
            return torch.zeros(self.num_envs, 3, 224, 224, device=self.device)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        rgb = rgb.permute(0, 3, 1, 2)
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0
        else:
            # Assume float input. Check range and normalize if needed.
            # Isaac Lab sometimes returns [-1, 1] or [0, 1] or [0, 255] float.
            # If we see negative values, likely [-1, 1] or zero-centered.
            if rgb.min() < 0.0:
                # Assuming [-1, 1] -> [0, 1]
                rgb = (rgb + 1.0) / 2.0
            elif rgb.max() > 1.1:
                # Assuming [0, 255] float
                rgb = rgb / 255.0

            # Clamp to safe [0, 1]
            rgb = torch.clamp(rgb, 0.0, 1.0)

        if rgb.shape[-2:] != (224, 224):
            rgb = torch.nn.functional.interpolate(
                rgb, size=(224, 224), mode="bilinear", align_corners=False
            )

        # --- MIRROR IMAGE ---
        # Flip Width (dim=3) to match Y-inversion or Camera Mirroring
        # rgb = torch.flip(rgb, [3])
        # --------------------

        return rgb

    def _process_proprio(self, obs):
        """Extracts and formats proprio from raw obs (Wrist)."""
        policy_obs = obs.get("policy", {})
        ee_pose = policy_obs.get("ee_pose", None)  # Expected (B, 7) from panda_hand
        gripper_pos = policy_obs.get("gripper_width", None)

        if ee_pose is None or gripper_pos is None:
            return torch.zeros(self.num_envs, self.proprio_dim, device=self.device)

        # 1. Wrist Data
        pos_wrist = ee_pose[:, :3]  # Meters
        quat_wxyz = ee_pose[:, 3:7]

        # Permute to xyzw for internal processing
        # Isaac Sim/Lab uses wxyz (scalar first)
        # Axis/PyPose uses xyzw (scalar last)
        quat_xyzw = torch.cat([quat_wxyz[:, 1:], quat_wxyz[:, 0:1]], dim=1)

        # PyPose SE3 for the wrist
        wrist_se3 = pp.SE3(torch.cat([pos_wrist, quat_xyzw], dim=1))

        # PyPose SE3 for the offset
        offset_pos = torch.tensor([[0.0, 0.0, 0.107]], device=self.device).repeat(pos_wrist.shape[0], 1)
        offset_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).repeat(pos_wrist.shape[0], 1)
        offset_se3 = pp.SE3(torch.cat([offset_pos, offset_quat], dim=1))

        # Apply offset to get tip pose
        tip_se3 = wrist_se3 @ offset_se3

        pos_tip = tip_se3.translation()
        
        # --- SCALING: METERS -> MILLIMETERS ---
        pos_tip_mm = pos_tip * 1000.0

        # Extract 9D Rotation Matrix and flatten it
        rot9 = tip_se3.rotation().matrix().flatten(start_dim=1)

        # Gripper — DROID convention: 1.0 = fully open, 0.0 = fully closed
        # Franka finger joints: 0.04m = fully open, 0.0m = fully closed
        # So g_meters / 0.04 directly maps to DROID convention (no inversion)
        g_meters = gripper_pos.mean(dim=1, keepdim=True)
        g_norm = g_meters / 0.04  # 0.04m (open) → 1.0, 0.0m (closed) → 0.0
        g_norm = torch.clamp(g_norm, 0.0, 1.0)

        # Output: [Rot9(9), Pos_mm(3), Gripper(1)] -> 13D
        proprio = torch.cat([rot9, pos_tip_mm, g_norm], dim=1)
        return proprio
