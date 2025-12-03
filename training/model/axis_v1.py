import torch
import torch.nn as nn
from .encoders import VisionEncoder, GoalEncoder, ProprioEncoder
from .token_learner import TokenLearner
from .latent import LatentQueue, LatentProjection
from .transformer import AxisTransformer
from .decoders import RequeryDecoder, ActionDecoder

class AxisModel(nn.Module):
    """
    The Axis V1 Model.
    Integrates Vision, Semantic Goals, and Latent Memory into a Transformer-based policy.
    Inputs:
        - Images: (B, Window, C, H, W)
        - Proprio: (B, Window, 7) [Pos, Rot, Gripper]
        - Goal: (B, 64) [Parsed Gemini Output]
    Outputs:
        - Action: (B, 7) [Next EE Pose]
        - Requery: (B, 1)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.get('device', 'cpu')

        # --- Shared Components ---
        self.goal_encoder = GoalEncoder(
            input_dim=config.get('goal_dim', 64), 
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

        # --- Robot Components (Directly Instantiated) ---
        self.proprio_encoder = ProprioEncoder(
            input_dim=config.get('proprio_dim', 7),
            output_dim=embed_dim,
            hidden_dim=config.get('hidden_dim', 128)
        )

        self.action_decoder = ActionDecoder(
            input_dim=embed_dim,
            output_dim=config.get('action_dim', 7),
            hidden_dim=config.get('hidden_dim', 128)
        )

    def forward(self, images, proprio, goal_embedding, update_queue=True, use_memory=False):
        """
        Args:
            images: (B, Window, C, H, W)
            proprio: (B, Window, proprio_dim)
            goal_embedding: (B, goal_dim)
            update_queue: bool
            use_memory: bool (Default False for early training)
        """
        B, W, C, H, _ = images.shape
        
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
        proprio_emb = self.proprio_encoder(proprio_flat) # (B*W, embed_dim)
        proprio_tokens = proprio_emb.reshape(B, W, 1, -1) # (B, W, 1, embed_dim)

        # 4. Construct Sequence: Interleaved [Proprio_t, Vision_t_1...Vision_t_k]
        # We want: P0, V0_1..V0_k, P1, ...
        # Concatenate along token dim then flatten
        # (B, W, 1+K, D)
        
        # vision_tokens from TokenLearner is (B*W, K, D)
        K = vision_tokens.shape[1]
        vision_tokens = vision_tokens.reshape(B, W, K, -1) # (B, W, K, embed_dim)
        
        context_tokens = torch.cat([proprio_tokens, vision_tokens], dim=2)
        context_tokens = context_tokens.reshape(B, W * (1 + K), -1) # (B, T_seq, D)

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
        # Window indices: 0..W-1.
        # Stride per step: 1 + K
        # Proprio indices: 0, 1+K, 2(1+K)...
        # Last Proprio index: (W-1) * (1 + K)
        stride = 1 + K
        readout_idx = (W - 1) * stride
        last_token = output_tokens[:, readout_idx, :]
        
        pred_action = self.action_decoder(last_token)
        
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

