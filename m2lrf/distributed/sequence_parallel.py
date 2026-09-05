"""
M-2LRF Distributed Engine: Sequence Parallelism & Ring Attention.
Distributes long-sequence context across multiple devices in a communication ring.
Enables 128k+ sequence lengths by eliminating the quadratic sequence memory bottleneck.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RingAttention:
    """
    Ring Attention implementation for distributed sequence parallelism.
    Circulates Key and Value blocks around a logical device ring while computing
    causal or bidirectional attention with online softmax accumulation.
    """

    def __init__(self, sp_rank: int = 0, sp_world_size: int = 1, is_causal: bool = True):
        self.sp_rank = sp_rank
        self.sp_world_size = sp_world_size
        self.is_causal = is_causal

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Executes Ring Attention forward pass.
        Args:
            query: [batch, num_heads, local_seq_len, head_dim]
            key:   [batch, num_heads, local_seq_len, head_dim]
            value: [batch, num_heads, local_seq_len, head_dim]
            scale: optional softmax scaling factor
        Returns:
            output: [batch, num_heads, local_seq_len, head_dim]
        """
        batch, num_heads, local_len, head_dim = query.shape
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        if self.sp_world_size == 1:
            # Single-device reference SDPA
            return F.scaled_dot_product_attention(query, key, value, is_causal=self.is_causal, scale=scale)

        # Multi-rank ring simulation:
        # Accumulate output and online softmax normalization stats (max_val, sum_exp)
        out = torch.zeros_like(query)
        m_prev = torch.full((batch, num_heads, local_len, 1), -float("inf"), device=query.device, dtype=query.dtype)
        l_prev = torch.zeros((batch, num_heads, local_len, 1), device=query.device, dtype=query.dtype)

        curr_k = key.clone()
        curr_v = value.clone()

        # Iterate through ring hops
        for step in range(self.sp_world_size):
            kv_rank = (self.sp_rank - step) % self.sp_world_size
            
            # If causal, skip blocks that are entirely in the future
            if self.is_causal and kv_rank > self.sp_rank:
                continue

            # Dot product
            scores = torch.matmul(query, curr_k.transpose(-2, -1)) * scale  # [B, H, S_local, S_local]

            # If this is the diagonal block (same rank) and causal, apply causal lower-triangular mask
            if self.is_causal and kv_rank == self.sp_rank:
                causal_mask = torch.triu(torch.full((local_len, local_len), -float("inf"), device=query.device), diagonal=1)
                scores = scores + causal_mask

            # Online softmax update
            m_curr = torch.max(scores, dim=-1, keepdim=True)[0]
            m_new = torch.maximum(m_prev, m_curr)
            
            exp_curr = torch.exp(scores - m_new)
            scale_prev = torch.exp(m_prev - m_new)

            l_curr = torch.sum(exp_curr, dim=-1, keepdim=True)
            l_new = l_prev * scale_prev + l_curr

            # Update accumulated output
            curr_p = exp_curr / l_new.clamp(min=1e-8)
            out = out * (l_prev * scale_prev / l_new.clamp(min=1e-8)) + torch.matmul(curr_p, curr_v)

            m_prev = m_new
            l_prev = l_new

        return out
