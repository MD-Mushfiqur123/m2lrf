"""
M-2LRF Multi-Modal Engine: 2-Bit Vision Transformer (ViT) Encoder.
===================================================================
Implements a complete Vision Transformer (SigLIP / CLIP / DINOv2 compatible)
using M-2LRF 2-bit dual-basis linear layers and patch embeddings.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.unified_layer import M2LRFUnifiedLinear


class PatchEmbeddings(nn.Module):
    """Splits image into non-overlapping patches and projects to hidden dimension."""

    def __init__(self, image_size: int = 224, patch_size: int = 14, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, C, H, W]
        Returns:
            patch_tokens: [B, NumPatches, EmbedDim]
        """
        x = self.proj(pixel_values)  # [B, EmbedDim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, NumPatches, EmbedDim]
        return x


class ViTAttention(nn.Module):
    """Multi-Head Self-Attention with 2-bit dual-basis projections."""

    def __init__(self, embed_dim: int, num_heads: int, bits: int = 2, rank: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = M2LRFUnifiedLinear(embed_dim, 3 * embed_dim, bits=bits, rank=rank)
        self.out_proj = M2LRFUnifiedLinear(embed_dim, embed_dim, bits=bits, rank=rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        return self.out_proj(attn)


class ViTMLP(nn.Module):
    """Feedforward network with GELU activation and 2-bit dual-basis layers."""

    def __init__(self, embed_dim: int, intermediate_dim: int, bits: int = 2, rank: int = 16):
        super().__init__()
        self.fc1 = M2LRFUnifiedLinear(embed_dim, intermediate_dim, bits=bits, rank=rank)
        self.act = nn.GELU()
        self.fc2 = M2LRFUnifiedLinear(intermediate_dim, embed_dim, bits=bits, rank=rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class ViTBlock(nn.Module):
    """Transformer Encoder Block with Pre-LayerNorm."""

    def __init__(self, embed_dim: int, num_heads: int, intermediate_dim: int, bits: int = 2, rank: int = 16):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ViTAttention(embed_dim, num_heads, bits=bits, rank=rank)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = ViTMLP(embed_dim, intermediate_dim, bits=bits, rank=rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerEncoder(nn.Module):
    """
    Complete Vision Transformer (ViT) Encoder.
    Outputs dense patch representations ready for multi-modal projection.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_dim: int = 3072,
        bits: int = 2,
        rank: int = 16,
    ):
        super().__init__()
        self.patch_embed = PatchEmbeddings(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, intermediate_dim, bits=bits, rank=rank)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, H, W]
        Returns:
            features: [B, NumPatches, EmbedDim]
        """
        x = self.patch_embed(pixel_values)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
