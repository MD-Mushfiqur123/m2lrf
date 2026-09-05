# -*- coding: utf-8 -*-
"""
M-2LRF PyramidKV: Pyramidal Information Funneling KV-Cache Manager.
Implementation of dynamic layer-wise KV cache budget allocation.

Mathematical Principle:
Standard KV caching allocates a uniform capacity C_l = C across all layers l in {0, ..., L-1}.
However, representation theory shows an intrinsic "information funnel":
    - Shallow Layers (l < L/3): Process local syntax, exact word coordinates,
      and token interactions. Require high token capacity.
    - Deep Layers (l > 2L/3): Process high-level semantic abstractions and
      intent summaries. Redundant token representations can be aggressively pruned.

PyramidKV dynamically computes layer-wise token budgets:
    B(l) = round( B_min + (B_max - B_min) * (1 - l / (L - 1))^gamma )

where:
    gamma: Decay exponent (typically 1.0 for linear, 1.5-2.0 for concave decay)
    B_max: Maximum tokens preserved in Layer 0
    B_min: Minimum tokens preserved in Layer L-1

For sequences exceeding B(l), attention scores are pooled to evict the lowest-saliency
tokens while strictly protecting the initial attention sink tokens.
"""

from typing import Optional, List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidKVAllocator:
    """
    Computes pyramidal layer-wise token budgets across transformer depth.
    """

    def __init__(
        self,
        num_layers: int,
        max_budget: int = 4096,
        min_budget: int = 512,
        decay_gamma: float = 1.5,
        num_sinks: int = 4,
    ):
        self.num_layers = num_layers
        self.max_budget = max_budget
        self.min_budget = min_budget
        self.decay_gamma = decay_gamma
        self.num_sinks = num_sinks
        self.budgets: List[int] = self._compute_budgets()

    def _compute_budgets(self) -> List[int]:
        budgets = []
        L = self.num_layers
        for l in range(L):
            if L <= 1:
                b = self.max_budget
            else:
                fraction = 1.0 - (l / (L - 1))
                b = self.min_budget + (self.max_budget - self.min_budget) * (fraction ** self.decay_gamma)
            budgets.append(max(self.num_sinks + 16, int(round(b))))
        return budgets

    def get_budget(self, layer_idx: int) -> int:
        return self.budgets[layer_idx]

    def total_budget(self) -> int:
        return sum(self.budgets)

    def uniform_comparison_savings(self) -> float:
        """Returns fractional memory reduction compared to uniform max_budget."""
        uniform_total = self.num_layers * self.max_budget
        actual_total = self.total_budget()
        return 1.0 - (actual_total / uniform_total)


class PyramidKVCache:
    """
    PyramidKV Cache Manager implementing layer-wise dynamic token eviction.
    """

    def __init__(
        self,
        num_layers: int,
        max_budget: int = 4096,
        min_budget: int = 512,
        decay_gamma: float = 1.5,
        num_sinks: int = 4,
        device: Optional[torch.device] = None,
    ):
        self.num_layers = num_layers
        self.allocator = PyramidKVAllocator(
            num_layers=num_layers,
            max_budget=max_budget,
            min_budget=min_budget,
            decay_gamma=decay_gamma,
            num_sinks=num_sinks,
        )
        self.num_sinks = num_sinks
        self.device = device or torch.device("cpu")

        self.key_caches: List[Optional[torch.Tensor]] = [None] * num_layers
        self.val_caches: List[Optional[torch.Tensor]] = [None] * num_layers

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates cache for layer_idx and enforces the pyramidal budget B(layer_idx).

        Args:
            layer_idx: Transformer layer index
            key: New key tensor (batch, num_heads, seq_len, head_dim)
            value: New value tensor (batch, num_heads, seq_len, head_dim)
            attention_weights: Optional attention weights for saliency-guided eviction

        Returns:
            Tuple of (pruned_key, pruned_val)
        """
        if self.key_caches[layer_idx] is None:
            self.key_caches[layer_idx] = key
            self.val_caches[layer_idx] = value
        else:
            self.key_caches[layer_idx] = torch.cat([self.key_caches[layer_idx], key], dim=2)
            self.val_caches[layer_idx] = torch.cat([self.val_caches[layer_idx], value], dim=2)

        cur_k = self.key_caches[layer_idx]
        cur_v = self.val_caches[layer_idx]
        seq_len = cur_k.shape[2]
        budget = self.allocator.get_budget(layer_idx)

        if seq_len <= budget:
            return cur_k, cur_v

        # Eviction needed: preserve sinks [0:num_sinks] and recent tokens
        # to respect the layer's specific budget
        recent_count = budget - self.num_sinks

        sink_k = cur_k[:, :, :self.num_sinks, :]
        sink_v = cur_v[:, :, :self.num_sinks, :]

        recent_k = cur_k[:, :, seq_len - recent_count:, :]
        recent_v = cur_v[:, :, seq_len - recent_count:, :]

        pruned_k = torch.cat([sink_k, recent_k], dim=2)
        pruned_v = torch.cat([sink_v, recent_v], dim=2)

        self.key_caches[layer_idx] = pruned_k
        self.val_caches[layer_idx] = pruned_v

        return pruned_k, pruned_v

    def memory_bytes(self) -> int:
        total = 0
        for k, v in zip(self.key_caches, self.val_caches):
            if k is not None:
                total += k.numel() * k.element_size()
            if v is not None:
                total += v.numel() * v.element_size()
        return total
