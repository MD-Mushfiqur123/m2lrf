# -*- coding: utf-8 -*-
"""
M-2LRF MiLoRA: Minor Singular Components Parameter-Efficient Adaptation.
Implementation of the 2024-2025 technique for preserving pre-trained knowledge
and preventing catastrophic forgetting during post-training fine-tuning.

Mathematical Rationale:
In foundation models, the singular spectrum of weight matrices W is heavy-tailed:
    W = U @ Σ @ V^T = ∑_{i=1}^d σ_i u_i v_i^T

The top-r principal singular components (i = 1, ..., r) encode universal features,
general syntax, and foundational pre-training representations.
Directly updating or perturbing these principal components often leads to:
    1. Catastrophic forgetting of previous tasks
    2. Degradation in out-of-distribution reasoning benchmarks
    3. Severe representation drift

MiLoRA identifies and targets the MINOR singular subspace (i = d - r + 1, ..., d):
    B_init = U_{minor} @ sqrt(Σ_{minor})
    A_init = sqrt(Σ_{minor}) @ V_{minor}^T
    W_residual = W - B_init @ A_init

By confining adapter optimization to the minor / null subspace:
    <u_minor, u_principal> = 0
    <v_minor, v_principal> = 0
The foundational knowledge remains strictly preserved with zero interference.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class M2LRFMiLoRALinear(nn.Module):
    """
    MiLoRA Linear Layer with M-2LRF integration.
    Initializes adapters on minor singular vectors of pretrained weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
        minor_offset: int = 0,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            r: Adapter rank
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout probability
            minor_offset: Optional shift from the absolute tail of singular values
            bias: Whether to include bias parameter
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.minor_offset = minor_offset

        # Frozen base weight
        self.register_buffer(
            "weight",
            torch.zeros((out_features, in_features), device=device, dtype=dtype or torch.float32),
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, device=device, dtype=dtype or torch.float32)
            )
        else:
            self.register_parameter("bias", None)

        if r > 0:
            self.lora_A = nn.Parameter(
                torch.zeros((r, in_features), device=device, dtype=dtype or torch.float32)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, r), device=device, dtype=dtype or torch.float32)
            )
            self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
            self.reset_parameters()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.dropout = nn.Identity()

        self.initialized_from_minor_svd = False

    def reset_parameters(self):
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    @torch.no_grad()
    def initialize_from_minor_svd(self, pretrained_weight: torch.Tensor):
        """
        Performs SVD on pretrained_weight and initializes lora_A and lora_B
        from the MINOR (tail) singular components.
        """
        if self.r <= 0:
            self.weight.copy_(pretrained_weight)
            return

        W_fp32 = pretrained_weight.float()
        # Compute full or economy SVD
        # U in R^{m x k}, S in R^{k}, Vh in R^{k x n}
        U, S, Vh = torch.linalg.svd(W_fp32, full_matrices=False)

        k = S.shape[0]
        actual_r = min(self.r, k)

        # Select minor components from the end of the spectrum
        start_idx = max(0, k - actual_r - self.minor_offset)
        end_idx = start_idx + actual_r

        U_minor = U[:, start_idx:end_idx]      # (m, r)
        S_minor = S[start_idx:end_idx]          # (r,)
        Vh_minor = Vh[start_idx:end_idx, :]    # (r, n)

        sqrt_S = torch.diag(torch.sqrt(S_minor))

        # B = U_minor @ sqrt(S)
        # A = sqrt(S) @ Vh_minor
        B_init = torch.matmul(U_minor, sqrt_S)
        A_init = torch.matmul(sqrt_S, Vh_minor)

        # Subtract minor adapter contribution from base weight
        W_residual = W_fp32 - torch.matmul(B_init, A_init)

        self.weight.copy_(W_residual.to(self.weight.dtype))
        self.lora_A.data.copy_(A_init.to(self.lora_A.dtype))
        self.lora_B.data.copy_(B_init.to(self.lora_B.dtype))

        self.initialized_from_minor_svd = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)

        if self.r <= 0:
            return base_out

        adapter_in = self.dropout(x)
        adapter_h = F.linear(adapter_in, self.lora_A)
        adapter_out = F.linear(adapter_h, self.lora_B)

        return base_out + self.scaling * adapter_out

    def merge(self) -> torch.Tensor:
        """Merges low-rank update back into base weights."""
        if self.r > 0:
            delta = (self.scaling * (self.lora_B @ self.lora_A)).to(self.weight.dtype)
            self.weight.add_(delta)
            self.lora_B.data.zero_()
            self.lora_A.data.zero_()
        return self.weight
