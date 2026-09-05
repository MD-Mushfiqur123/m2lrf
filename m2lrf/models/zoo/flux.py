"""
M-2LRF Model Zoo: Flux.1 (12B Rectified Flow Multimodal Diffusion Transformer).
==============================================================================
Implements Flux.1 MMDiT with M-2LRF 2-bit dual-basis linear layers:
- Dual-stream blocks: separate processing of image latents and text conditioning with cross-attention
- Single-stream blocks: concatenated processing of fused image-text tokens
- 2-bit quantization across all query, key, value, and MLP projections
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.unified_layer import M2LRFUnifiedLinear


class FluxConfig:
    def __init__(
        self,
        hidden_size: int = 3072,
        num_heads: int = 24,
        head_dim: int = 128,
        num_dual_layers: int = 19,
        num_single_layers: int = 38,
        mlp_ratio: float = 4.0,
        bits: int = 2,
        rank: int = 16,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_dual_layers = num_dual_layers
        self.num_single_layers = num_single_layers
        self.mlp_ratio = mlp_ratio
        self.bits = bits
        self.rank = rank


class DualStreamBlock(nn.Module):
    """Dual-stream MMDiT block: separate image and text pathways with joint attention."""

    def __init__(self, config: FluxConfig):
        super().__init__()
        dim = config.hidden_size
        mlp_dim = int(dim * config.mlp_ratio)

        # Image stream
        self.img_norm1 = nn.LayerNorm(dim)
        self.img_qkv = M2LRFUnifiedLinear(dim, 3 * dim, bits=config.bits, rank=config.rank)
        self.img_norm2 = nn.LayerNorm(dim)
        self.img_mlp = nn.Sequential(
            M2LRFUnifiedLinear(dim, mlp_dim, bits=config.bits, rank=config.rank),
            nn.GELU(),
            M2LRFUnifiedLinear(mlp_dim, dim, bits=config.bits, rank=config.rank),
        )

        # Text stream
        self.txt_norm1 = nn.LayerNorm(dim)
        self.txt_qkv = M2LRFUnifiedLinear(dim, 3 * dim, bits=config.bits, rank=config.rank)
        self.txt_norm2 = nn.LayerNorm(dim)
        self.txt_mlp = nn.Sequential(
            M2LRFUnifiedLinear(dim, mlp_dim, bits=config.bits, rank=config.rank),
            nn.GELU(),
            M2LRFUnifiedLinear(mlp_dim, dim, bits=config.bits, rank=config.rank),
        )

        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

    def forward(self, img: torch.Tensor, txt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = img.shape[0]
        # Joint QKV computation
        iq, ik, iv = torch.chunk(self.img_qkv(self.img_norm1(img)), 3, dim=-1)
        tq, tk, tv = torch.chunk(self.txt_qkv(self.txt_norm1(txt)), 3, dim=-1)

        # Concatenate keys and values
        k = torch.cat([tk, ik], dim=1).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = torch.cat([tv, iv], dim=1).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        iq = iq.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        tq = tq.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        img_attn = F.scaled_dot_product_attention(iq, k, v).transpose(1, 2).contiguous().view(bsz, -1, self.num_heads * self.head_dim)
        txt_attn = F.scaled_dot_product_attention(tq, k, v).transpose(1, 2).contiguous().view(bsz, -1, self.num_heads * self.head_dim)

        img = img + img_attn + self.img_mlp(self.img_norm2(img))
        txt = txt + txt_attn + self.txt_mlp(self.txt_norm2(txt))

        return img, txt


class SingleStreamBlock(nn.Module):
    """Single-stream block processing unified sequence of concatenated image and text tokens."""

    def __init__(self, config: FluxConfig):
        super().__init__()
        dim = config.hidden_size
        mlp_dim = int(dim * config.mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.qkv = M2LRFUnifiedLinear(dim, 3 * dim, bits=config.bits, rank=config.rank)
        self.mlp = nn.Sequential(
            M2LRFUnifiedLinear(dim, mlp_dim, bits=config.bits, rank=config.rank),
            nn.GELU(),
            M2LRFUnifiedLinear(mlp_dim, dim, bits=config.bits, rank=config.rank),
        )
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        norm_x = self.norm(x)
        q, k, v = torch.chunk(self.qkv(norm_x), 3, dim=-1)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return x + attn + self.mlp(norm_x)


class FluxTransformer(nn.Module):
    """Complete Flux.1 Diffusion Transformer Model with 2-bit dual-basis weights."""

    def __init__(self, config: FluxConfig):
        super().__init__()
        self.config = config
        self.dual_blocks = nn.ModuleList([DualStreamBlock(config) for _ in range(config.num_dual_layers)])
        self.single_blocks = nn.ModuleList([SingleStreamBlock(config) for _ in range(config.num_single_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, img_latents: torch.Tensor, txt_conditioning: torch.Tensor) -> torch.Tensor:
        img = img_latents
        txt = txt_conditioning

        # 1. Dual stream stages
        for block in self.dual_blocks:
            img, txt = block(img, txt)

        # 2. Single stream stage
        fused = torch.cat([txt, img], dim=1)
        for block in self.single_blocks:
            fused = block(fused)

        return self.final_norm(fused[:, txt.shape[1] :])  # Return image latent prediction
