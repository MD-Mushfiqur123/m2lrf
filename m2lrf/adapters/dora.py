"""
M-2LRF Weight-Decomposed Low-Rank Adaptation (DoRA) (PEFT-Inspired)
====================================================================
Decouples magnitude and directional updates over 2-bit base weights:
W = m * (W_0 + (alpha/r)*B@A) / ||W_0 + (alpha/r)*B@A||_c
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.unified_layer import M2LRF2BitLinear


class M2LRFDoRALinear(nn.Module):
    """
    DoRA layer wrapping 2-bit dual-basis base weights with decomposed magnitude vector.
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
            rank=0,  # Rank in base layer is 0; DoRA manages its own adapter
            bias=bias,
            group_size=group_size
        )

        # DoRA adapters
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Learnable magnitude vector: [out_features, 1]
        self.magnitude = nn.Parameter(torch.ones(out_features, 1))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def initialize_from_pretrained(self, weight: torch.Tensor):
        self.base_layer.initialize_from_pretrained(weight)
        # Initialize magnitude from column norm of base weight
        with torch.no_grad():
            col_norm = weight.norm(p=2, dim=1, keepdim=True)
            self.magnitude.data.copy_(col_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Dequantize base weights
        w_dequant = self.base_layer._dequantize_base(dtype=x.dtype)  # [out_features, in_features]

        # 2. Add LoRA directional update: V = W_0 + (alpha/r)*B@A
        lora_delta = (self.lora_B @ self.lora_A) * self.scaling
        v = w_dequant + lora_delta

        # 3. Normalize direction: V_norm = V / ||V||_c
        v_norm = v / v.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)

        # 4. Scale by magnitude vector: W_eff = m * V_norm
        w_eff = self.magnitude * v_norm

        # 5. Compute GEMM
        out = F.linear(x, w_eff, self.bias)
        return out
