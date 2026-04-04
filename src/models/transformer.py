import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) as introduced in:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)

    RoPE encodes position information by rotating the query and key vectors,
    enabling relative position awareness without explicit position embeddings.
    """

    def __init__(self, dim, max_seq_len=512, base=10000):
        """
        Args:
            dim: Per-head dimension (embed_dim // num_heads)
            max_seq_len: Maximum sequence length to precompute
            base: Base for the frequency computation
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute inverse frequencies: theta_i = base^(-2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin for efficiency
        self._precompute_cache(max_seq_len)

    def _precompute_cache(self, seq_len):
        """Precompute the cos and sin values for the given sequence length."""
        positions = torch.arange(seq_len, device=self.inv_freq.device).float()
        # Shape: (seq_len, dim/2)
        freqs = torch.outer(positions, self.inv_freq)
        # Create full rotation matrix by repeating for each pair
        # Shape: (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len, device):
        """
        Returns cos and sin position encodings for the given sequence length.

        Args:
            seq_len: Length of the sequence
            device: Device to place tensors on

        Returns:
            Tuple of (cos, sin) each with shape (seq_len, dim)
        """
        if seq_len > self.max_seq_len:
            # Dynamically extend cache if needed
            self._precompute_cache(seq_len)
            self.max_seq_len = seq_len

        return (
            self.cos_cached[:seq_len].to(device),
            self.sin_cached[:seq_len].to(device),
        )


def rotate_half(x):
    """
    Rotates half the hidden dims of the input.
    Used in applying rotary embeddings.

    Args:
        x: Tensor of shape (..., dim)
    Returns:
        Tensor with rotated dimensions of shape (..., dim)
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, num_heads, seq_len, head_dim)
        cos: Cosine position encodings of shape (seq_len, head_dim)
        sin: Sine position encodings of shape (seq_len, head_dim)

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs
    """
    # Reshape cos and sin for broadcasting: (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Apply rotation using the formula:
    # q_rot = q * cos + rotate_half(q) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class RoPEMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Rotary Position Embeddings.

    This is a custom implementation that applies RoPE to Q and K
    before computing attention scores.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, max_seq_len=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        self.scale = self.head_dim**-0.5

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            attn_mask: Optional attention mask
            key_padding_mask: Optional key padding mask

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        B, S, D = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: (B, S, D) -> (B, num_heads, S, head_dim)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Get rotary embeddings and apply to Q, K
        cos, sin = self.rope(S, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask if provided
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: (B, S) -> (B, 1, 1, S)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: (B, num_heads, S, head_dim) -> (B, S, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)

        # Output projection
        output = self.out_proj(attn_output)

        return output


class RoPETransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with RoPE attention.

    Follows the standard Pre-LN transformer architecture:
    x -> LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual
    """

    def __init__(
        self, embed_dim, num_heads, dim_feedforward, dropout=0.1, max_seq_len=512
    ):
        super().__init__()

        # Self-attention with RoPE
        self.self_attn = RoPEMultiHeadAttention(
            embed_dim, num_heads, dropout=dropout, max_seq_len=max_seq_len
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout),
        )

        # Layer norms (Pre-LN architecture)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            src_mask: Optional source mask for attention
            src_key_padding_mask: Optional padding mask

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        # Pre-LN Self-Attention
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        x = self.dropout(x) + residual

        # Pre-LN FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x) + residual

        return x


class RoPETransformerEncoder(nn.Module):
    """
    Stack of RoPE Transformer Encoder Layers.
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                encoder_layer if i == 0 else self._clone_layer(encoder_layer)
                for i in range(num_layers)
            ]
        )
        self.norm = norm
        self.num_layers = num_layers

    def _clone_layer(self, layer):
        """Create a new layer with the same configuration."""
        return RoPETransformerEncoderLayer(
            embed_dim=layer.self_attn.embed_dim,
            num_heads=layer.self_attn.num_heads,
            dim_feedforward=layer.ffn[0].out_features,
            dropout=layer.dropout.p,
            max_seq_len=layer.self_attn.rope.max_seq_len,
        )

    def forward(self, x, mask=None, src_key_padding_mask=None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            mask: Optional attention mask
            src_key_padding_mask: Optional padding mask

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        for layer in self.layers:
            x = layer(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


class AxisTransformer(nn.Module):
    """
    Main Transformer backbone for the Axis model.
    Processes a sequence of tokens: [LatentHistory, VisionTokens, ProprioTokens].

    Supports two positional embedding modes:
    - RoPE (default): Rotary Position Embeddings applied to Q/K in attention
    - Learned: Traditional learned positional embeddings added to input
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
        use_rope=True,
        max_seq_len=128,
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            use_rope: If True, use Rotary Position Embeddings; else use learned embeddings
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len

        if use_rope:
            # Use custom transformer with RoPE
            encoder_layer = RoPETransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )
            self.transformer = RoPETransformerEncoder(
                encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(embed_dim)
            )
            # No positional embedding needed - RoPE handles it
            self.pos_embedding = None
        else:
            # Use standard PyTorch transformer with learned positional embeddings
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )

            # Learnable positional embeddings
            self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

    def forward(self, x):
        """
        Args:
            x: Input tokens (B, SeqLen, EmbedDim)
        Returns:
            Output tokens (B, SeqLen, EmbedDim)
        """
        B, S, D = x.shape

        if not self.use_rope and self.pos_embedding is not None:
            # Add learned positional embeddings (truncated to current sequence length)
            x = x + self.pos_embedding[:, :S, :]

        # Pass through transformer
        # RoPE is applied internally in the attention layers when use_rope=True
        x = self.transformer(x)

        return x
