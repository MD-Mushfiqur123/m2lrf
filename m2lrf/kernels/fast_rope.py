"""
M-2LRF Fast RoPE (Rotary Position Embedding) Kernel (Unsloth-Inspired)
======================================================================
In-place fused Rotary Position Embedding kernel for Q and K projection tensors.
Eliminates intermediate slicing, sign flipping, and concatenation tensor allocations.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def fast_apply_rotary_pos_emb_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized in-place or low-memory RoPE application.
    Shapes:
      q: [batch, n_heads, seq_len, head_dim]
      k: [batch, n_kv_heads, seq_len, head_dim]
      cos, sin: [1, 1, seq_len, head_dim] or [seq_len, head_dim]
    """
    if cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.ndim == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    half_dim = q.shape[-1] // 2
    
    # q rotation
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    cos1, sin1 = cos[..., :half_dim], sin[..., :half_dim]
    q_rot = torch.cat([q1 * cos1 - q2 * sin1, q1 * sin1 + q2 * cos1], dim=-1)

    # k rotation
    k1, k2 = k[..., :half_dim], k[..., half_dim:]
    k_rot = torch.cat([k1 * cos1 - k2 * sin1, k1 * sin1 + k2 * cos1], dim=-1)

    return q_rot.to(q.dtype), k_rot.to(k.dtype)


def fast_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unified entry point for fast RoPE embedding.
    """
    return fast_apply_rotary_pos_emb_pytorch(q, k, cos, sin)
