"""
M-2LRF Model Zoo: Command-R+ (Cohere Enterprise RAG Architecture).
==================================================================
Implements Command-R+ with M-2LRF 2-bit dual-basis linear layers:
- LayerNorm with learnable bias
- Grouped Query Attention (GQA) with 64 attention heads and 8 KV heads
- Wide intermediate expansion with SwiGLU feed-forward networks
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.unified_layer import M2LRFUnifiedLinear


class CommandRConfig:
    def __init__(
        self,
        vocab_size: int = 256000,
        hidden_size: int = 8192,
        intermediate_size: int = 22528,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        bits: int = 2,
        group_size: int = 64,
        rank: int = 16,
        layer_norm_eps: float = 1e-5,
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
        self.layer_norm_eps = layer_norm_eps


class CommandRAttention(nn.Module):
    def __init__(self, config: CommandRConfig):
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

        if self.num_heads != self.num_kv_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(out)


class CommandRDecoderLayer(nn.Module):
    def __init__(self, config: CommandRConfig):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = CommandRAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gate_proj = M2LRFUnifiedLinear(config.hidden_size, config.intermediate_size, bits=config.bits, rank=config.rank)
        self.up_proj = M2LRFUnifiedLinear(config.hidden_size, config.intermediate_size, bits=config.bits, rank=config.rank)
        self.down_proj = M2LRFUnifiedLinear(config.intermediate_size, config.hidden_size, bits=config.bits, rank=config.rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        mlp = self.down_proj(F.silu(self.gate_proj(self.post_attention_layernorm(x))) * self.up_proj(self.post_attention_layernorm(x)))
        return x + mlp


class CommandRForCausalLM(nn.Module):
    """Complete Cohere Command-R+ Causal Language Model with 2-bit dual-basis weights."""

    def __init__(self, config: CommandRConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([CommandRDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(self.final_layer_norm(h))
