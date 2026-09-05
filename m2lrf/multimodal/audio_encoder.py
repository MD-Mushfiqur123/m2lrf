"""
M-2LRF Multi-Modal Engine: 2-Bit Audio & Speech Encoder.
========================================================
Implements a Whisper / Conformer style acoustic encoder:
1. 1D Convolutional downsampling front-end (reduces 10ms acoustic frames).
2. Sinusoidal positional embeddings.
3. Multi-head self-attention with 2-bit dual-basis linear layers.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.unified_layer import M2LRFUnifiedLinear


class AudioConvSubsampler(nn.Module):
    """Downsamples log-mel spectrogram frames using two 1D/2D convolutions with stride 2."""

    def __init__(self, in_features: int = 80, out_features: int = 512):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, out_features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(out_features, out_features, kernel_size=3, stride=2, padding=1)
        self.act = nn.GELU()

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spectrogram: [B, NumMels, TimeFrames]
        Returns:
            subsampled: [B, TimeFrames // 2, OutFeatures]
        """
        x = self.act(self.conv1(mel_spectrogram))
        x = self.act(self.conv2(x))
        return x.transpose(1, 2)  # [B, T', C]


class AudioAttention(nn.Module):
    """Acoustic Self-Attention with 2-bit dual-basis projections."""

    def __init__(self, embed_dim: int, num_heads: int, bits: int = 2, rank: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = M2LRFUnifiedLinear(embed_dim, embed_dim, bits=bits, rank=rank)
        self.k_proj = M2LRFUnifiedLinear(embed_dim, embed_dim, bits=bits, rank=rank)
        self.v_proj = M2LRFUnifiedLinear(embed_dim, embed_dim, bits=bits, rank=rank)
        self.out_proj = M2LRFUnifiedLinear(embed_dim, embed_dim, bits=bits, rank=rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        return self.out_proj(attn)


class AudioEncoderBlock(nn.Module):
    """Transformer Encoder Block with Pre-LayerNorm for acoustic representations."""

    def __init__(self, embed_dim: int, num_heads: int, intermediate_dim: int, bits: int = 2, rank: int = 16):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = AudioAttention(embed_dim, num_heads, bits=bits, rank=rank)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = M2LRFUnifiedLinear(embed_dim, intermediate_dim, bits=bits, rank=rank)
        self.act = nn.GELU()
        self.fc2 = M2LRFUnifiedLinear(intermediate_dim, embed_dim, bits=bits, rank=rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.fc2(self.act(self.fc1(self.norm2(x))))
        return x


class AudioTransformerEncoder(nn.Module):
    """Complete 2-Bit Audio & Speech Acoustic Encoder."""

    def __init__(
        self,
        in_mels: int = 80,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_dim: int = 2048,
        bits: int = 2,
        rank: int = 16,
    ):
        super().__init__()
        self.subsampler = AudioConvSubsampler(in_features=in_mels, out_features=embed_dim)
        self.blocks = nn.ModuleList([
            AudioEncoderBlock(embed_dim, num_heads, intermediate_dim, bits=bits, rank=rank)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spectrogram: [B, NumMels, TimeFrames]
        Returns:
            acoustic_features: [B, TimeFrames // 2, EmbedDim]
        """
        x = self.subsampler(mel_spectrogram)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
