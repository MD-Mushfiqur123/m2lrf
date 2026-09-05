"""
M-2LRF Low-Rank Hadamard Product (LoHa) Adapter (LyCORIS-Inspired)
===================================================================
Expresses weight updates via the Hadamard product of two low-rank matrices:
Delta W = (alpha / rank) * (B_1 @ A_1) * (B_2 @ A_2)
Provides effective rank up to r^2 with parameter footprint 4 * r * d.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.unified_layer import M2LRF2BitLinear


class M2LRFLoHaLinear(nn.Module):
    """
    LoHa layer combining 2-bit dual-basis base weights with dual low-rank Hadamard product.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        bias: bool = False,
        group_size: Optional[int] = 128
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0

        # Base 2-bit layer
        self.base_layer = M2LRF2BitLinear(
            in_features=in_features,
            out_features=out_features,
            rank=0,
            bias=bias,
            group_size=group_size
        )

        # LoHa matrices: W_1 = B1 @ A1, W_2 = B2 @ A2
        self.A1 = nn.Parameter(torch.empty(rank, in_features))
        self.B1 = nn.Parameter(torch.zeros(out_features, rank))
        self.A2 = nn.Parameter(torch.empty(rank, in_features))
        self.B2 = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.kaiming_uniform_(self.A1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.A2, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def initialize_from_pretrained(self, weight: torch.Tensor):
        self.base_layer.initialize_from_pretrained(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)

        # Compute Delta W = (B1 @ A1) * (B2 @ A2)
        w1 = self.B1 @ self.A1
        w2 = self.B2 @ self.A2
        delta_w = (w1 * w2) * self.scaling

        loha_out = F.linear(x, delta_w)
        return base_out + loha_out
