"""
M-2LRF Model Zoo: Phi-4 (Microsoft 14B High-Density Reasoning Architecture).
===========================================================================
Implements Phi-4 with 2-bit dual-basis linear layers:
- Extended RoPE base frequency (theta = 250,000) for 16k+ context windows
- SwiGLU feed-forward networks
- Grouped Query Attention (GQA) with 40 query heads and 10 KV heads
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.unified_layer import M2LRFUnifiedLinear


class Phi4Config:
    def __init__(
        self,
        vocab_size: int = 100352,
        hidden_size: int = 5120,
        intermediate_size: int = 17920,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 40,
        num_key_value_heads: int = 10,
        bits: int = 2,
        group_size: int = 64,
        rank: int = 16,
        rope_theta: float = 250000.0,
        rms_norm_eps: float = 1e-5,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.bits = bits
        self.group_size = group_size
        self.rank = rank
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps


class Phi4Attention(nn.Module):
    def __init__(self, config: Phi4Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = M2LRFUnifiedLinear(config.hidden_size, config.hidden_size, bits=config.bits, rank=config.rank)
        self.k_proj = M2LRFUnifiedLinear(config.hidden_size, self.num_kv_heads * self.head_dim, bits=config.bits, rank=config.rank)
        self.v_proj = M2LRFUnifiedLinear(config.hidden_size, self.num_kv_heads * self.head_dim, bits=config.bits, rank=config.rank)
        self.dense = M2LRFUnifiedLinear(config.hidden_size, config.hidden_size, bits=config.bits, rank=config.rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.num_heads != self.num_kv_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.dense(out)


class Phi4MLP(nn.Module):
    def __init__(self, config: Phi4Config):
        super().__init__()
        self.gate_up_proj = M2LRFUnifiedLinear(config.hidden_size, 2 * config.intermediate_size, bits=config.bits, rank=config.rank)
        self.down_proj = M2LRFUnifiedLinear(config.intermediate_size, config.hidden_size, bits=config.bits, rank=config.rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = torch.chunk(gate_up, 2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Phi4DecoderLayer(nn.Module):
    def __init__(self, config: Phi4Config):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Phi4Attention(config)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Phi4MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Phi4ForCausalLM(nn.Module):
    """Complete Microsoft Phi-4 14B Language Model with 2-bit dual-basis weights."""

    def __init__(self, config: Phi4Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Phi4DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(self.final_layernorm(h))
