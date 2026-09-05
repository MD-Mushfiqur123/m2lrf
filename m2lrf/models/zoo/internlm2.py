"""
M-2LRF Native 2-Bit Architecture: InternLM2
Configured with True 2-Bit Dual-Basis Linear Projections and LoftQ Initialization.
"""

from typing import Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.unified_layer import M2LRFUnifiedLinear


class InternLM2Config:
    def __init__(
        self,
        vocab_size: int = 92544,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        max_position_embeddings: int = 8192,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 500000.0,
        bits: int = 2,
        group_size: int = 64,
        rank: int = 16,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.bits = bits
        self.group_size = group_size
        self.rank = rank


class InternLM2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class InternLM2RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, S, 1, D]
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class InternLM2Attention(nn.Module):
    def __init__(self, config: InternLM2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Native 2-bit dual-basis projections
        self.q_proj = M2LRFUnifiedLinear(
            in_features=self.hidden_size,
            out_features=self.num_heads * self.head_dim,
            bits=config.bits,
            group_size=config.group_size,
            rank=config.rank,
        )
        self.k_proj = M2LRFUnifiedLinear(
            in_features=self.hidden_size,
            out_features=self.num_key_value_heads * self.head_dim,
            bits=config.bits,
            group_size=config.group_size,
            rank=config.rank,
        )
        self.v_proj = M2LRFUnifiedLinear(
            in_features=self.hidden_size,
            out_features=self.num_key_value_heads * self.head_dim,
            bits=config.bits,
            group_size=config.group_size,
            rank=config.rank,
        )
        self.o_proj = M2LRFUnifiedLinear(
            in_features=self.num_heads * self.head_dim,
            out_features=self.hidden_size,
            bits=config.bits,
            group_size=config.group_size,
            rank=config.rank,
        )
        self.rotary_emb = InternLM2RotaryEmbedding(self.head_dim, base=config.rope_theta)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = self.rotary_emb(v, seq_len=q_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Transpose to [B, H, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_out)


class InternLM2MLP(nn.Module):
    def __init__(self, config: InternLM2Config):
        super().__init__()
        self.gate_proj = M2LRFUnifiedLinear(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            bits=config.bits,
            group_size=config.group_size,
            rank=config.rank,
        )
        self.up_proj = M2LRFUnifiedLinear(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            bits=config.bits,
            group_size=config.group_size,
            rank=config.rank,
        )
        self.down_proj = M2LRFUnifiedLinear(
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            bits=config.bits,
            group_size=config.group_size,
            rank=config.rank,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class InternLM2DecoderLayer(nn.Module):
    def __init__(self, config: InternLM2Config):
        super().__init__()
        self.self_attn = InternLM2Attention(config)
        self.mlp = InternLM2MLP(config)
        self.input_layernorm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class InternLM2Model(nn.Module):
    def __init__(self, config: InternLM2Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([InternLM2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)


class InternLM2ForCausalLM(nn.Module):
    def __init__(self, config: InternLM2Config):
        super().__init__()
        self.config = config
        self.model = InternLM2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            return loss, logits

        return logits
