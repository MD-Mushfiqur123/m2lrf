"""
M-2LRF PiSSA: Principal Singular Values & Vectors Adaptation (ICLR 2024 / GraphPKU)
====================================================================================
Initializes LoRA adapters with the top-r principal singular components of the pretrained
weight matrix W_0, while quantizing the low-energy residual matrix into M-2LRF 2-bit dual-basis.

Mathematical Formulation:
  W_0 = U * Sigma * V^T
  Adapter B = U[:, :r] * sqrt(Sigma[:r])
  Adapter A = sqrt(Sigma[:r]) * V[:r, :]
  Residual W_res = W_0 - B @ A  (Quantized to 2-bit Dual-Basis)

Convergence Advantage:
  Unlike standard LoRA (random A, zero B), PiSSA directly captures the most energetic
  spectral subspace, accelerating fine-tuning convergence by 2x-4x.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.unified_layer import M2LRFUnifiedLinear


class M2LRFPiSSALinear(nn.Module):
    """
    PiSSA linear layer with 2-bit dual-basis residual storage.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: Optional[float] = None,
        bias: bool = False,
        group_size: Optional[int] = 128,
        use_hadamard: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = float(alpha) if alpha is not None else float(rank)
        self.scaling = self.alpha / rank if rank > 0 else 1.0

        # Base 2-bit layer storing the residual W_res
        self.base_layer = M2LRFUnifiedLinear(
            in_features=in_features,
            out_features=out_features,
            bits=2,
            rank=0,  # Adapter is managed directly by PiSSA
            bias=bias,
            group_size=group_size,
            use_hadamard=use_hadamard
        )

        # Principal adapters
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def initialize_from_pretrained(self, weight: torch.Tensor):
        """
        Extracts top-r singular components into lora_A and lora_B,
        and quantizes the residual matrix into 2-bit dual-basis.
        """
        orig_dtype = weight.dtype
        w_fp32 = weight.float()

        # Perform truncated SVD
        # U: [out_features, k], S: [k], Vh: [k, in_features]
        try:
            U, S, Vh = torch.linalg.svd(w_fp32, full_matrices=False)
        except Exception:
            # Fallback for CPU / LAPACK stability
            U, S, V = torch.svd(w_fp32)
            Vh = V.t()

        Ur = U[:, :self.rank]
        Sr = S[:self.rank]
        Vhr = Vh[:self.rank, :]

        sqrt_S = torch.sqrt(Sr)

        # B = Ur * sqrt(S) / sqrt(scaling)
        # A = sqrt(S) * Vhr / sqrt(scaling)
        scale_factor = math.sqrt(self.scaling)
        self.lora_B.data.copy_((Ur * sqrt_S.unsqueeze(0)) / scale_factor)
        self.lora_A.data.copy_((sqrt_S.unsqueeze(1) * Vhr) / scale_factor)

        # Compute residual: W_res = W_0 - (alpha/r) * B @ A
        principal_approx = (self.lora_B @ self.lora_A) * self.scaling
        residual = w_fp32 - principal_approx

        # Quantize residual to 2-bit dual-basis
        self.base_layer.initialize_from_pretrained(residual.to(orig_dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base residual forward (2-bit packed)
        base_out = self.base_layer(x)

        # Principal component forward
        adapter_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling

        return base_out + adapter_out
