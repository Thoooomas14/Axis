import torch
import gymnasium as gym
import isaaclab.utils.math as math_utils

def _quat_to_rot6d(quat_xyzw):
    """
    Convert (B, 4) quaternion [x,y,z,w] to (B, 6) rotation 6D.
    Inlined to avoid cross-module dependency issues in Isaac Sim extensions.
    """
    # Normalize
    quat = quat_xyzw / (torch.norm(quat_xyzw, dim=-1, keepdim=True) + 1e-8)
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # First column of rotation matrix
    r00 = 1 - 2 * (y * y + z * z)
    r10 = 2 * (x * y + z * w)
    r20 = 2 * (x * z - y * w)
    
    # Second column
    r01 = 2 * (x * y - z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r21 = 2 * (y * z + x * w)
    
    # Stack columns (B, 6)
    return torch.stack([r00, r10, r20, r01, r11, r21], dim=-1)

class AxisObservationWrapper(gym.Wrapper):
    """
    Wraps the Isaac Lab environment to provide history-windowed observations
    compatible with the Axis Transformer Model.
    
    Returns:
        images: (B, Window, C, H, W)
        proprio: (B, Window, 10) -> [Pos(3), Rot6D(6), Gripper(1)]
    """
    def __init__(self, env, window_size=8, device="cpu"):
        super().__init__(env)
        self.window_size = window_size
        self.device = device
        self.image_buffer = None
        self.proprio_buffer = None
        self.num_envs = getattr(env, "num_envs", 1)
        self.proprio_dim = 13 # Axis V2 uses 13D

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if 'policy' not in obs: pass
        img_current = self._process_image(obs)
        prop_current = self._process_proprio(obs)
        B = img_current.shape[0]
        C, H, W = img_current.shape[1], img_current.shape[2], img_current.shape[3]
        
        # Initialize buffers
        self.image_buffer = torch.zeros(B, self.window_size, C, H, W, device=self.device)
        self.proprio_buffer = torch.zeros(B, self.window_size, self.proprio_dim, device=self.device)
        
        # Fill with current observation (repeat for initial window)
        self.image_buffer[:] = img_current.unsqueeze(1)
        self.proprio_buffer[:] = prop_current.unsqueeze(1)
        
        return self._get_obs(), info

    def step(self, action, model_run = True):
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
            'images': self.image_buffer.clone(), 
            'proprio': self.proprio_buffer.clone() 
        }

    def _process_image(self, obs):
        rgb = obs.get('policy', {}).get('rgb', None)
        if rgb is None: return torch.zeros(self.num_envs, 3, 128, 128, device=self.device)
        if rgb.shape[-1] == 4: rgb = rgb[..., :3]
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

        if rgb.shape[-2:] != (128, 128):
             rgb = torch.nn.functional.interpolate(rgb, size=(128, 128), mode='bilinear', align_corners=False)
        
        # --- MIRROR IMAGE ---
        # Flip Width (dim=3) to match Y-inversion or Camera Mirroring
        # rgb = torch.flip(rgb, [3])
        # --------------------

        return rgb

    def _process_proprio(self, obs):
        """Extracts and formats proprio from raw obs (Wrist)."""
        policy_obs = obs.get('policy', {})
        ee_pose = policy_obs.get('ee_pose', None) # Expected (B, 7) from panda_hand
        gripper_pos = policy_obs.get('gripper_width', None)
        
        if ee_pose is None or gripper_pos is None:
            return torch.zeros(self.num_envs, self.proprio_dim, device=self.device)
            
        # 1. Wrist Data
        pos_wrist = ee_pose[:, :3] # Meters
        quat_wxyz = ee_pose[:, 3:7] 
        
        # --- CALCULATE TIP POSITION ---
        # Apply offset (0.107m in Z) to Wrist Frame
        offset = torch.tensor([0.0, 0.0, 0.107], device=self.device).repeat(pos_wrist.shape[0], 1)
        pos_delta = math_utils.quat_apply(quat_wxyz, offset)
        pos_tip = pos_wrist + pos_delta # Meters
        
        # --- MIRROR PROPRIO Y (Tip Frame) ---
        # pos_tip[:, 1] = -pos_tip[:, 1]
        # ------------------------
        
        # --- SCALING: METERS -> MILLIMETERS ---
        pos_tip_mm = pos_tip * 1000.0
        
        # Permute to xyzw for internal processing
        # Isaac Sim/Lab uses wxyz (scalar first)
        # Axis/PyPose uses xyzw (scalar last)
        quat_xyzw = torch.cat([quat_wxyz[:, 1:], quat_wxyz[:, 0:1]], dim=1)
        
        # Convert to Rotation Matrix (9D)
        # Using math_utils if available or torch ops
        # math_utils.quat_to_rot_matrix likely exists but we can use our helper or simple logic
        # Isaac Lab math_utils uses wxyz usually.
        # Let's use robust manual computation from xyzw to avoid dependency issues within wrapper
        x, y, z, w = quat_xyzw[..., 0], quat_xyzw[..., 1], quat_xyzw[..., 2], quat_xyzw[..., 3]
        
        r00 = 1 - 2 * (y*y + z*z)
        r01 = 2 * (x*y - z*w)
        r02 = 2 * (x*z + y*w)
        
        r10 = 2 * (x*y + z*w)
        r11 = 1 - 2 * (x*x + z*z)
        r12 = 2 * (y*z - x*w)
        
        r20 = 2 * (x*z - y*w)
        r21 = 2 * (y*z + x*w)
        r22 = 1 - 2 * (x*x + y*y)
        
        # Stack to (B, 9) flat
        rot9 = torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1)
        
        # Gripper
        g_meters = gripper_pos.mean(dim=1, keepdim=True)
        g_norm = 1.0 - (g_meters / 0.04)
        g_norm = torch.clamp(g_norm, 0.0, 1.0)
        
        # Output: [Rot9(9), Pos_mm(3), Gripper(1)] -> 13D
        proprio = torch.cat([rot9, pos_tip_mm, g_norm], dim=1)
        return proprio
