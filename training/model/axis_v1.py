import torch
import torch.nn as nn
from .encoders import VisionEncoder, GoalEncoder
from .token_learner import TokenLearner
from .latent import LatentQueue, LatentProjection
from .transformer import AxisTransformer
from .decoders import RequeryDecoder
from .adapters import RobotAdapter

class AxisModel(nn.Module):
    """
    The Axis V1 Model (Multi-Robot Support).
    Integrates Vision, Semantic Goals, and Latent Memory into a Transformer-based policy.
    Uses RobotAdapters to handle robot-specific proprioception and actions.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.get('device', 'cpu')

        # --- Shared Components ---
        self.goal_encoder = GoalEncoder(
            input_dim=config.get('goal_dim', 768), 
            output_dim=config.get('cond_dim', 256)
        )
        
        self.vision_encoder = VisionEncoder(
            cond_dim=config.get('cond_dim', 256)
        )
        
        # --- Bottleneck ---
        self.token_learner = TokenLearner(
            input_channels=config.get('vision_feature_dim', 256),
            num_tokens=config.get('num_vision_tokens', 8)
        )

        # --- Latent Memory ---
        self.latent_dim = config.get('latent_dim', 256)
        self.latent_queue = LatentQueue(
            queue_size=config.get('queue_size', 10),
            latent_dim=self.latent_dim,
            device=self.device
        )
        self.latent_proj = LatentProjection(
            latent_dim=self.latent_dim,
            embed_dim=config.get('embed_dim', 256)
        )

        # --- Backbone ---
        self.transformer = AxisTransformer(
            embed_dim=config.get('embed_dim', 256),
            num_heads=config.get('num_heads', 4),
            num_layers=config.get('num_layers', 4)
        )
        
        # --- Latent Output ---
        # We use a simple linear projection (or Identity) to avoid "garbage" from a deep MLP
        # without supervision. This forces the transformer output to be directly useful as memory.
        embed_dim = config.get('embed_dim', 256)
        if self.latent_dim == embed_dim:
            self.latent_proj_out = nn.Identity()
        else:
            self.latent_proj_out = nn.Linear(embed_dim, self.latent_dim)
        
        self.requery_decoder = RequeryDecoder(
            embed_dim=embed_dim
        )

        # --- Robot Adapters ---
        # config['robots'] should be a dict: { 'robot_name': { 'proprio_dim': 6, 'action_dim': 6 } }
        self.adapters = nn.ModuleDict()
        if 'robots' in config:
            for robot_name, robot_cfg in config['robots'].items():
                self.adapters[robot_name] = RobotAdapter(
                    proprio_input_dim=robot_cfg['proprio_dim'],
                    action_output_dim=robot_cfg['action_dim'],
                    embed_dim=config.get('embed_dim', 256)
                )
        else:
            # Fallback for backward compatibility or single robot mode
            # We create a 'default' adapter
            self.adapters['default'] = RobotAdapter(
                proprio_input_dim=config.get('proprio_dim', 6),
                action_output_dim=config.get('action_dim', 6),
                embed_dim=config.get('embed_dim', 256)
            )

    def forward(self, images, proprio, goal_embedding, robot_name='default', update_queue=True):
        """
        Args:
            images: (B, C, H, W)
            proprio: (B, proprio_dim)
            goal_embedding: (B, goal_dim)
            robot_name: str, name of the robot to use (must be in self.adapters)
            update_queue: Whether to update the latent queue (inference mode)
        Returns:
            pred_action: (B, action_dim)
            next_latent: (B, latent_dim)
            requery_logit: (B, 1)
        """
        B = images.shape[0]
        
        if robot_name not in self.adapters:
            raise ValueError(f"Robot '{robot_name}' not found in adapters: {list(self.adapters.keys())}")
        
        adapter = self.adapters[robot_name]

        # 1. Encode Goal
        goal_cond = self.goal_encoder(goal_embedding) # (B, cond_dim)

        # 2. Encode Vision (Conditioned on Goal)
        vision_features = self.vision_encoder(images, goal_cond) # (B, C_feat, H', W')
        vision_tokens = self.token_learner(vision_features) # (B, K, embed_dim)

        # 3. Encode Proprio (Robot Specific)
        proprio_emb = adapter.encode_proprio(proprio) # (B, embed_dim)
        proprio_tokens = proprio_emb.unsqueeze(1) # (B, 1, embed_dim)

        # 4. Get Latent History
        latent_history = self.latent_queue.forward()
        if latent_history.shape[0] != B:
            latent_history = latent_history.expand(B, -1, -1)
            
        latent_tokens = self.latent_proj(latent_history) # (B, T, embed_dim)

        # 5. Concatenate Tokens
        input_tokens = torch.cat([latent_tokens, vision_tokens, proprio_tokens], dim=1)

        # 6. Transformer Pass
        output_tokens = self.transformer(input_tokens)

        # 7. Decode
        last_token = output_tokens[:, -1, :]
        
        pred_action = adapter.decode_action(last_token) # Robot Specific
        
        # Simple projection for latent state (Identity if dims match)
        next_latent = self.latent_proj_out(last_token)
        
        requery_logit = self.requery_decoder(last_token)

        # 8. Update Queue (Inference only usually)
        if update_queue:
            self.latent_queue.enqueue(next_latent.detach())

        return pred_action, next_latent, requery_logit
