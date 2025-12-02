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

    def forward(self, images, proprio, goal_embedding, robot_name='default', update_queue=True, use_memory=True):
        """
        Args:
            images: (B, Window, C, H, W)
            proprio: (B, Window, proprio_dim)
            goal_embedding: (B, goal_dim)
            robot_name: str
            update_queue: bool
            use_memory: bool
        """
        B, W, C, H, _ = images.shape
        
        if robot_name not in self.adapters:
            raise ValueError(f"Robot '{robot_name}' not found in adapters: {list(self.adapters.keys())}")
        
        adapter = self.adapters[robot_name]

        # 1. Encode Goal (Shared across window)
        goal_cond = self.goal_encoder(goal_embedding) # (B, cond_dim)
        # Expand goal for the window batch processing
        goal_cond_expanded = goal_cond.unsqueeze(1).expand(-1, W, -1).reshape(B*W, -1)

        # 2. Encode Vision
        # Flatten B and W -> (B*W, C, H, W)
        images_flat = images.reshape(B*W, C, H, -1)
        vision_features = self.vision_encoder(images_flat, goal_cond_expanded) # (B*W, C_feat, H', W')
        vision_tokens = self.token_learner(vision_features) # (B*W, 1, embed_dim)
        vision_tokens = vision_tokens.reshape(B, W, 1, -1).squeeze(2) # (B, W, embed_dim)

        # 3. Encode Proprio
        proprio_flat = proprio.reshape(B*W, -1)
        proprio_emb = adapter.encode_proprio(proprio_flat) # (B*W, embed_dim)
        proprio_tokens = proprio_emb.reshape(B, W, -1) # (B, W, embed_dim)

        # 4. Construct Sequence: Interleaved [Proprio_t, Vision_t]
        # We want: P0, V0, P1, V1, ...
        # Stack along new dim then flatten
        # (B, W, 2, D)
        context_tokens = torch.stack([proprio_tokens, vision_tokens], dim=2)
        context_tokens = context_tokens.reshape(B, W*2, -1) # (B, 2*W, D)

        # 5. Add Latent History (Optional)
        # Latent is appended at the end
        input_tokens_list = [context_tokens]
        
        if use_memory:
            latent_history = self.latent_queue.forward()
            if latent_history.shape[0] != B:
                latent_history = latent_history.expand(B, -1, -1)
            latent_tokens = self.latent_proj(latent_history) # (B, T_latent, embed_dim)
            input_tokens_list.append(latent_tokens)

        input_tokens = torch.cat(input_tokens_list, dim=1)

        # 6. Transformer Pass
        output_tokens = self.transformer(input_tokens)

        # 7. Decode
        # Readout from the LAST Proprio token in the window.
        # Window indices: 0..2W-1.
        # Proprio indices: 0, 2, 4...
        # Last Proprio index: (W-1)*2
        # Example W=8: Indices 0..15. Last Proprio is 14. Last Vision is 15.
        # We want to predict action for step t (current), which corresponds to the last frame in window.
        # So we read from the Proprio token of the LAST step.
        readout_idx = (W - 1) * 2
        last_token = output_tokens[:, readout_idx, :]
        
        pred_action = adapter.decode_action(last_token) # Robot Specific
        
        # Simple projection for latent state
        next_latent = self.latent_proj_out(last_token)
        
        requery_logit = self.requery_decoder(last_token)

        # 8. Update Queue
        if update_queue:
            self.latent_queue.enqueue(next_latent.detach())

        return pred_action, next_latent, requery_logit

    def reset_memory(self, batch_size):
        """Resets the latent memory for a new batch."""
        self.latent_queue.reset(batch_size)
