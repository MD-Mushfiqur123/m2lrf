"""
M-2LRF Multi-Modal Engine: High-Performance Cross-Modal Projectors.
===================================================================
Provides modular projectors bridging vision/audio encoders to LLM embedding spaces:
1. LinearProjector: Simple projection with 2-bit dual-basis weights.
2. MLPProjector: 2-layer GELU MLP with residual connections (LLaVA-1.5 / Qwen-VL style).
3. PerceiverResampler: Cross-attention perceiver resampler compressing arbitrary visual
   tokens into a fixed number of latent visual queries (Flamingo / IDEFICS style).
4. PixelShuffleProjector: Spatial 2D pixel unshuffle downsampler compressing 4x spatial tokens.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.unified_layer import M2LRFUnifiedLinear


class LinearProjector(nn.Module):
    """Simple linear projection from visual/audio feature dimension to LLM hidden dimension."""

    def __init__(self, in_features: int, out_features: int, bits: int = 2, rank: int = 16):
        super().__init__()
        self.proj = M2LRFUnifiedLinear(
            in_features=in_features,
            out_features=out_features,
            bits=bits,
            rank=rank,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MLPProjector(nn.Module):
    """Two-layer GELU MLP projector widely used in modern vision-language models."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: Optional[int] = None,
        bits: int = 2,
        rank: int = 16,
    ):
        super().__init__()
        hidden_dim = hidden_dim or out_features
        self.linear1 = M2LRFUnifiedLinear(
            in_features=in_features,
            out_features=hidden_dim,
            bits=bits,
            rank=rank,
        )
        self.act = nn.GELU()
        self.linear2 = M2LRFUnifiedLinear(
            in_features=hidden_dim,
            out_features=out_features,
            bits=bits,
            rank=rank,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.act(self.linear1(x)))


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler using learned latent queries and multi-head cross-attention.
    Compresses variable-length visual tokens into a fixed number of visual tokens (num_latents).
    """

    def __init__(
        self,
        visual_dim: int,
        llm_dim: int,
        num_latents: int = 64,
        num_heads: int = 8,
        bits: int = 2,
        rank: int = 16,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.num_heads = num_heads
        self.head_dim = llm_dim // num_heads

        # Learned latent query vectors
        self.latents = nn.Parameter(torch.randn(num_latents, llm_dim) * 0.02)

        # Cross-attention projections
        self.q_proj = M2LRFUnifiedLinear(llm_dim, llm_dim, bits=bits, rank=rank)
        self.k_proj = M2LRFUnifiedLinear(visual_dim, llm_dim, bits=bits, rank=rank)
        self.v_proj = M2LRFUnifiedLinear(visual_dim, llm_dim, bits=bits, rank=rank)
        self.out_proj = M2LRFUnifiedLinear(llm_dim, llm_dim, bits=bits, rank=rank)

        # Layer norms
        self.norm_latents = nn.LayerNorm(llm_dim)
        self.norm_context = nn.LayerNorm(visual_dim)
        self.norm_out = nn.LayerNorm(llm_dim)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: [BatchSize, NumVisualTokens, VisualDim]
        Returns:
            compressed_tokens: [BatchSize, NumLatents, LLMDim]
        """
        bsz = visual_features.shape[0]
        # Expand latents across batch
        latents = self.latents.unsqueeze(0).expand(bsz, -1, -1)  # [B, NumLatents, LLMDim]

        normed_q = self.norm_latents(latents)
        normed_kv = self.norm_context(visual_features)

        q = self.q_proj(normed_q).view(bsz, self.num_latents, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed_kv).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed_kv).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot product cross-attention
        attn_out = F.scaled_dot_product_attention(q, k, v)  # [B, H, NumLatents, HeadDim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, self.num_latents, -1)

        out = self.out_proj(attn_out) + latents
        return self.norm_out(out)


class PixelShuffleProjector(nn.Module):
    """
    Downsamples 2D spatial feature maps via pixel unshuffle (space-to-channel)
    followed by a 2-bit linear projection.
    Reduces visual token count by 4x (e.g. from 576 tokens to 144 tokens).
    """

    def __init__(self, in_features: int, out_features: int, downsample_factor: int = 2, bits: int = 2, rank: int = 16):
        super().__init__()
        self.downsample_factor = downsample_factor
        unshuffled_dim = in_features * (downsample_factor ** 2)
        self.proj = M2LRFUnifiedLinear(
            in_features=unshuffled_dim,
            out_features=out_features,
            bits=bits,
            rank=rank,
        )

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Args:
            x: [B, H * W, C]
            height: spatial height
            width: spatial width
        Returns:
            downsampled: [B, (H/2)*(W/2), OutFeatures]
        """
        bsz, seq_len, c = x.shape
        assert seq_len == height * width, "Sequence length must match height * width"
        x = x.view(bsz, height, width, c)

        # Pixel unshuffle: reshape and permute
        f = self.downsample_factor
        h_down = height // f
        w_down = width // f
        x = x.view(bsz, h_down, f, w_down, f, c)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(bsz, h_down * w_down, f * f * c)

        return self.proj(x)
