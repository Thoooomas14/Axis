import torch
import torch.nn as nn
from .encoders import VisionEncoder, ProprioEncoder, GoalEncoder
from .transformer import AxisTransformer
from .decoders import ActionDecoder, RequeryDecoder, SafeActionDecoder


class AxisModel(nn.Module):
    """
    The Axis Model.
    
    Uses a concatenated single token per timestep: [Proprio(128) | Vision(256) | Goal(128)] -> 512 dim.
    Predicts future trajectory from the LAST token only (Current State -> Future).
    
    Inputs:
        - images: (B, W, C, H, W) - window of RGB images
        - proprio: (B, W, 13) - window of 13D SE(3) poses
        - goal: (B, 38) - standardized goal vector
    
    Outputs:
        - action: (B, ChunkSize, 7) - predicted future trajectory
        - requery: (B, 1) - prediction confidence
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Dimensions
        self.vision_dim = 256
        self.proprio_dim = 128
        self.goal_dim = 128
        self.embed_dim = self.vision_dim + self.proprio_dim + self.goal_dim # 512
        
        # --- Encoders ---
        
        self.goal_encoder = GoalEncoder(
            input_dim=config.get('goal_dim', 64),
            output_dim=self.goal_dim
        )
        
        self.vision_encoder = VisionEncoder(
            feature_dim=self.vision_dim,
            goal_dim=config.get('goal_dim', 38)
        )
        
        self.proprio_encoder = ProprioEncoder(
            input_dim=config.get('proprio_dim', 13),
            output_dim=self.proprio_dim,
            hidden_dim=config.get('hidden_dim', 128)
        )
        
        # --- Backbone ---
        self.transformer = AxisTransformer(
            embed_dim=self.embed_dim,
            num_heads=config.get('num_heads', 4), # Increased heads for 512 dim
            num_layers=config.get('num_layers', 4),
            use_rope=config.get('use_rope', True)
        )
        
        # --- Decoders ---
        action_dim = config.get('action_dim', 7)
        chunk_size = config.get('chunk_size', 10)
        
        base_decoder = ActionDecoder(
            input_dim=self.embed_dim,
            output_dim=action_dim,
            hidden_dim=config.get('hidden_dim', 128),
            chunk_size=chunk_size
        )
        
        self.action_decoder = SafeActionDecoder(base_decoder)
        self.requery_decoder = RequeryDecoder(embed_dim=self.embed_dim)
    
    def forward(self, images, proprio, raw_goal, cached_tokens=None, return_tokens=False):
        """
        Forward pass for Axis V2.
        
        Args:
            images: (B, W, C, H, W) - window of RGB images
            proprio: (B, W, 10) - window of 10D proprioceptive states
            raw_goal: (B, 38) or (B, W, 38) - standardized raw goal vector
            cached_tokens: (B, W-1, 512) - optional cached tokens from previous step
            return_tokens: bool - whether to return input tokens for caching
            
        Returns:
            pred_action: (B, 10) or (B, chunk_size, 10) - predicted action(s)
            requery_logit: (B, 1) - subtask completion logit
            tokens: (B, W, 512) - (Optional) full token sequence if return_tokens=True
        """
        B, W, C, H, _ = images.shape
        
        if cached_tokens is not None and cached_tokens.shape[1] == W - 1:
            # OPTIMIZED PATH: Process only the NEWEST frame (index -1)
            
            # Extract NEWEST raw goal first to pass to VisionEncoder
            if raw_goal.dim() == 3:
                goal_raw_new = raw_goal[:, -1, :] # (B, 38)
            else:
                goal_raw_new = raw_goal # (B, 38) assumed constant
                
            # Encode NEWEST goal for transformer backbone
            goal_feat_new = self.goal_encoder(goal_raw_new) # (B, 128)
            goal_tokens_new = goal_feat_new.unsqueeze(1) # (B, 1, 128)
            
            # Encode NEWEST vision frame (passing RAW goal)
            # images slice: (B, 1, C, H, W) -> flatten to (B*1, C, H, W)
            images_new = images[:, -1:, ...].reshape(B, C, H, -1)
            vision_tokens_new = self.vision_encoder(images_new, raw_goal=goal_raw_new) # pass RAW goal!
            vision_tokens_new = vision_tokens_new.unsqueeze(1) # (B, 1, 256)
            
            # Encode NEWEST proprio frame
            proprio_new = proprio[:, -1:, ...].reshape(B, -1) # (B, 10)
            proprio_tokens_new = self.proprio_encoder(proprio_new) # (B, 128)
            proprio_tokens_new = proprio_tokens_new.unsqueeze(1) # (B, 1, 128)
            
            # Concatenate for the new frame
            new_tokens = torch.cat([proprio_tokens_new, vision_tokens_new, goal_tokens_new], dim=-1) # (B, 1, 512)
            
            # Combine with cache
            input_tokens = torch.cat([cached_tokens, new_tokens], dim=1) # (B, W, 512)
            
        else:
            # STANDARD PATH: Process FULL window
            
            # 1. Process Raw Goal & Encode for Backbone
            if raw_goal.dim() == 3:
                # Need raw goal sequence for vision encoder
                goal_raw_flat = raw_goal.reshape(B*W, -1) # (B*W, 38)
                
                # Encode for backbone
                goal_tokens_flat = self.goal_encoder(goal_raw_flat) # (B*W, 128)
                goal_tokens = goal_tokens_flat.reshape(B, W, 128)
            else:
                # Raw goal is constant across window
                goal_raw_flat = raw_goal.repeat_interleave(W, dim=0) # (B*W, 38)
                
                # Encode for backbone
                goal_feat = self.goal_encoder(raw_goal) # (B, 128)
                goal_tokens = goal_feat.unsqueeze(1).expand(B, W, -1) # (B, W, 128)
        
            # 2. Encode vision (passing RAW goal)
            images_flat = images.reshape(B*W, C, H, -1)
            vision_tokens = self.vision_encoder(images_flat, raw_goal=goal_raw_flat)  # (B*W, 256)
            vision_tokens = vision_tokens.reshape(B, W, 256)  # (B, W, 256)
            
            # 3. Encode proprio
            proprio_flat = proprio.reshape(B*W, -1)
            proprio_tokens = self.proprio_encoder(proprio_flat)  # (B*W, 128)
            proprio_tokens = proprio_tokens.reshape(B, W, 128)   # (B, W, 128)
            
            # 4. Concatenate: [Proprio, Vision, Goal]
            input_tokens = torch.cat([proprio_tokens, vision_tokens, goal_tokens], dim=-1) # (B, W, 512)
        
        # 5. Transformer backbone
        output_tokens = self.transformer(input_tokens)  # (B, W, 512)
        
        # 6. Decode from LAST token ONLY (Current State -> Future Trajectory)
        last_token = output_tokens[:, -1, :]  # (B, 512)
        
        pred_action = self.action_decoder(last_token)  # (B, ChunkSize, 7)
        requery_logit = self.requery_decoder(last_token)  # (B, 1)
        
        if return_tokens:
            return pred_action, requery_logit, input_tokens
        else:
            return pred_action, requery_logit
