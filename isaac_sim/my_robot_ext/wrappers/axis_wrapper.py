import torch
import gymnasium as gym

class AxisObservationWrapper(gym.Wrapper):
    """
    Wraps the Isaac Lab environment to provide history-windowed observations
    compatible with the Axis Transformer Model.
    
    Output format:
        images: (B, Window, C, H, W)
        proprio: (B, Window, 8)
    """
    def __init__(self, env, window_size=8, device="cpu"):
        super().__init__(env)
        self.window_size = window_size
        self.device = device
        
        # Buffer initialization happens on reset
        self.image_buffer = None
        self.proprio_buffer = None
        self.num_envs = getattr(env, "num_envs", 1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Initialize buffers (assume obs contains 'image' and 'proprio' or we extract them)
        # We need to inspect 'obs' structure to know where to pull data from.
        # For now, we assume the underlying env (AxisEvalEnv) populates:
        #   obs['policy']['rgb'] -> (B, H, W, 3) or (B, H, W, 4)
        #   obs['policy']['proprio'] -> (B, 8)
        
        # Dummy init if keys missing (for safety during dev)
        if 'policy' not in obs:
             # Fallback or error
             pass
             
        # Extract initial frame
        # Standardize Image: (B, C, H, W)
        # obs['policy']['rgb'] is likely (B, H, W, 3)
        # We need to permute to (B, 3, H, W)
        img_current = self._process_image(obs)
        prop_current = self._process_proprio(obs)
        
        B = img_current.shape[0]
        C, H, W = img_current.shape[1], img_current.shape[2], img_current.shape[3]
        P_dim = prop_current.shape[1]
        
        # Create buffers: (B, Window, ...)
        self.image_buffer = torch.zeros(B, self.window_size, C, H, W, device=self.device)
        self.proprio_buffer = torch.zeros(B, self.window_size, P_dim, device=self.device)
        
        # Fill buffer with initial state (repeat or zeros? usually repeat for start)
        self.image_buffer[:] = img_current.unsqueeze(1)
        self.proprio_buffer[:] = prop_current.unsqueeze(1)
        
        return self._get_obs(), info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        
        img_current = self._process_image(obs) # (B, C, H, W)
        prop_current = self._process_proprio(obs) # (B, P_dim)
        
        # Shift buffer
        # [0, 1, 2, 3] -> [1, 2, 3, new]
        self.image_buffer = torch.roll(self.image_buffer, shifts=-1, dims=1)
        self.image_buffer[:, -1] = img_current
        
        self.proprio_buffer = torch.roll(self.proprio_buffer, shifts=-1, dims=1)
        self.proprio_buffer[:, -1] = prop_current
        
        return self._get_obs(), rew, terminated, truncated, info

    def _get_obs(self):
        """Returns the dictionary expected by AxisModel."""
        return {
            'images': self.image_buffer.clone(), # (B, W, C, H, W)
            'proprio': self.proprio_buffer.clone() # (B, W, 8)
        }

    def _process_image(self, obs):
        """Extracts and formats image from raw obs."""
        # Adjust key access based on actual EvalEnv output
        # Assuming obs['policy']['rgb'] exists
        rgb = obs.get('policy', {}).get('rgb', None)
        
        if rgb is not None:
            # Debug Stats Once
            if not hasattr(self, "_debug_printed_rgb"):
                print(f"[Wrapper] Raw RGB: Dtype={rgb.dtype} | Scale={rgb.min().item()}-{rgb.max().item()} | Shape={rgb.shape}")
                self._debug_printed_rgb = True
        
        if rgb is None:
            # Fallback for testing/debugging
            return torch.zeros(self.num_envs, 3, 128, 128, device=self.device)
            
        # Isaac Lab Images are often (B, H, W, 4) (RGBA)
        # We need RGB, (B, 3, H, W)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
            
        # Permute (B, H, W, C) -> (B, C, H, W)
        rgb = rgb.permute(0, 3, 1, 2)
        
        # Normalize 0-1 if uint8
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0
            
        return rgb

    def _process_proprio(self, obs):
        """Extracts and formats proprio from raw obs."""
        # Config defines:
        #   ee_pose: (B, 7)
        #   gripper_width: (B, 2) (usually 2 fingers)
        
        policy_obs = obs.get('policy', {})
        
        # --- DEBUG: Check Observation Keys ---
        # Print once (using a flag or just simplistic check if we haven't printed)
        if not hasattr(self, "_debug_printed_obs"):
            print(f"[Wrapper] Policy Obs Keys: {list(policy_obs.keys())}")
            for k, v in policy_obs.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.shape} | Mean: {v.float().mean():.3f} | Max: {v.max():.3f}")
            self._debug_printed_obs = True
        # -------------------------------------
        
        ee_pose = policy_obs.get('ee_pose', None)
        gripper_pos = policy_obs.get('gripper_width', None)
        
        if ee_pose is None or gripper_pos is None:
            # Fallback if keys missing
            return torch.zeros(self.num_envs, 8, device=self.device)
            
        # Isaac Lab ee_pose is (B, 7): [x, y, z, w, x, y, z] (Scalar First)
        # Model expects (B, 7): [x, y, z, x, y, z, w] (Scalar Last - Scipy)
        
        pos = ee_pose[:, :3]
        quat_wxyz = ee_pose[:, 3:7] # w, x, y, z
        
        # Permute to xyzw
        # w=0, x=1, y=2, z=3
        quat_xyzw = torch.cat([quat_wxyz[:, 1:], quat_wxyz[:, 0:1]], dim=1)
        
        # Gripper: take mean of 2 fingers to get 1D width-like value
        # Or Just take one. Usually they are symmetric.
        # (B, 2) -> (B, 1)
        # Raw meters: 0.0 (Closed) -> 0.04 (Open)
        g_meters = gripper_pos.mean(dim=1, keepdim=True)
        
        # Normalize to 0 (Open) -> 1 (Closed)
        # 1.0 - (0.04 / 0.04) = 0.0 (Open)
        # 1.0 - (0.00 / 0.04) = 1.0 (Closed)
        g_norm = 1.0 - (g_meters / 0.04)
        g_norm = torch.clamp(g_norm, 0.0, 1.0)
        
        # Concat: (B, 7) + (B, 1) -> (B, 8)
        # Order: [Pos(3), Quat(4, xyzw), Gripper(1)]
        proprio = torch.cat([pos, quat_xyzw, g_norm], dim=1)
            
        return proprio
