"""
M-2LRF Model Zoo: Qwen2-VL (Vision-Language Foundation Architecture).
====================================================================
Implements Qwen2-VL with 3D Rotary Position Embeddings (temporal, height, width)
and 2-bit dual-basis quantized linear layers.
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.unified_layer import M2LRFUnifiedLinear
from m2lrf.multimodal.vision_encoder import VisionTransformerEncoder
from m2lrf.multimodal.projectors import MLPProjector


class Qwen2VLConfig:
    def __init__(
        self,
        vocab_size: int = 152064,
        hidden_size: int = 3584,
        intermediate_size: int = 18944,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 28,
        num_key_value_heads: int = 4,
        vision_dim: int = 1152,
        image_size: int = 448,
        patch_size: int = 14,
        bits: int = 2,
        group_size: int = 64,
        rank: int = 16,
        rms_norm_eps: float = 1e-6,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.vision_dim = vision_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.bits = bits
        self.group_size = group_size
        self.rank = rank
        self.rms_norm_eps = rms_norm_eps


class Qwen2VLAttention(nn.Module):
    def __init__(self, config: Qwen2VLConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = M2LRFUnifiedLinear(config.hidden_size, config.hidden_size, bits=config.bits, rank=config.rank)
        self.k_proj = M2LRFUnifiedLinear(config.hidden_size, self.num_kv_heads * self.head_dim, bits=config.bits, rank=config.rank)
        self.v_proj = M2LRFUnifiedLinear(config.hidden_size, self.num_kv_heads * self.head_dim, bits=config.bits, rank=config.rank)
        self.o_proj = M2LRFUnifiedLinear(config.hidden_size, config.hidden_size, bits=config.bits, rank=config.rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # GQA repeat
        if self.num_heads != self.num_kv_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(out)


class Qwen2VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2VLConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Qwen2VLAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gate_proj = M2LRFUnifiedLinear(config.hidden_size, config.intermediate_size, bits=config.bits, rank=config.rank)
        self.up_proj = M2LRFUnifiedLinear(config.hidden_size, config.intermediate_size, bits=config.bits, rank=config.rank)
        self.down_proj = M2LRFUnifiedLinear(config.intermediate_size, config.hidden_size, bits=config.bits, rank=config.rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        mlp_out = self.down_proj(F.silu(self.gate_proj(self.norm2(x))) * self.up_proj(self.norm2(x)))
        return x + mlp_out


class Qwen2VLForConditionalGeneration(nn.Module):
    """Full Qwen2-VL Model combining ViT Vision Encoder, MLP Projector, and Language Model."""

    def __init__(self, config: Qwen2VLConfig):
        super().__init__()
        self.config = config
        self.visual = VisionTransformerEncoder(
            image_size=config.image_size,
            patch_size=config.patch_size,
            embed_dim=config.vision_dim,
            num_layers=4,
            bits=config.bits,
            rank=config.rank,
        )
        self.projector = MLPProjector(
            in_features=config.vision_dim,
            out_features=config.hidden_size,
            bits=config.bits,
            rank=config.rank,
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen2VLDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)

        if pixel_values is not None:
            vis_features = self.visual(pixel_values)
            proj_vis = self.projector(vis_features)
            # Concatenate visual tokens at start
            inputs_embeds = torch.cat([proj_vis, inputs_embeds], dim=1)

        h = inputs_embeds
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(self.norm(h))
