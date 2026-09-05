"""
M-2LRF QuaRot: Dual-Sided Orthogonal Incoherence Engine (NeurIPS 2024)
========================================================================
Applies randomized orthogonal Hadamard transformations across the residual stream
and projection weights:
  X_rot = X @ H
  W_rot = H^T @ W
  Output = X_rot @ W_rot = (X @ H) @ (H^T @ W) = X @ (H @ H^T) @ W = X @ W

Eliminates both activation outliers and weight outliers across Attention and MLP blocks
with zero mathematical loss under infinite precision.
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.hadamard_transform import fast_walsh_hadamard_transform, block_fast_walsh_hadamard_transform


class QuaRotLinear(nn.Module):
    """
    Linear layer with dual-sided orthogonal rotation for outlier-free activation & weight quantization.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int = 64,
        bias: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.rotated = False

    @torch.no_grad()
    def rotate_weights(self):
        """
        Rotates weight matrix in-place: W_rot = W @ H.
        """
        if not self.rotated:
            w_rot = block_fast_walsh_hadamard_transform(self.weight.data, block_size=self.block_size)
            self.weight.data.copy_(w_rot)
            self.rotated = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotates activations and computes linear GEMM: (x @ H) @ W_rot^T
        """
        # 1. Rotate activations
        x_rot = block_fast_walsh_hadamard_transform(x, block_size=self.block_size)

        # 2. GEMM with rotated weights
        out = F.linear(x_rot, self.weight, self.bias)
        return out
