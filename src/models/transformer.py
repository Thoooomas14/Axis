import torch
import torch.nn as nn

class AxisTransformer(nn.Module):
    """
    Main Transformer backbone for the Axis model.
    Processes a sequence of tokens: [LatentHistory, VisionTokens, ProprioTokens].
    """
    def __init__(self, embed_dim=256, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Transformer Encoder
        # We use batch_first=True for easier handling of (B, S, D) tensors
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learnable positional embeddings? 
        # Since the sequence structure is fixed [Latent... Vision... Proprio...], 
        # we might not strictly need them if the model can learn from position, 
        # but it's good practice.
        # We'll assume a max sequence length.
        self.max_seq_len = 128 # Should be enough for queue + tokens
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_seq_len, embed_dim))

    def forward(self, x):
        """
        Args:
            x: Input tokens (B, SeqLen, EmbedDim)
        Returns:
            Output tokens (B, SeqLen, EmbedDim)
        """
        B, S, D = x.shape
        
        # Add positional embeddings (truncated to current sequence length)
        x = x + self.pos_embedding[:, :S, :]
        
        # Pass through transformer
        x = self.transformer(x)
        
        return x
