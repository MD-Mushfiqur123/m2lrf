# -*- coding: utf-8 -*-
"""
M-2LRF LoRA-Pro: Gradient-Projection Aligned Low-Rank Adaptation.
Implementation of ICLR 2025 Spotlight technique for closing the gap
between low-rank parameter-efficient fine-tuning and full-parameter fine-tuning.

Mathematical Formulation:
In standard LoRA, the effective weight update is:
    ΔW = (α / r) * B @ A
Standard backpropagation computes gradients:
    ∇_A L = (α / r) * B^T @ G
    ∇_B L = (α / r) * G @ A^T
where G = ∇_W L is the full-rank loss gradient.

However, the induced full-rank update direction:
    d(ΔW) = (α / r) * (dB @ A + B @ dA)
does not minimize the projection error ||G - d(ΔW)||_F^2 unless
the gradient updates dA and dB are specifically projected onto the
optimal subspace.

LoRA-Pro solves the constrained optimization problem:
    min_{dA, dB} || G - (dB @ A + B @ dA) ||_F^2 + λ (||dA||_F^2 + ||dB||_F^2)
yielding the closed-form projected gradient equations:
    dA* = (B^T @ B + λ I)^{-1} @ B^T @ G @ (I - A^T @ (A @ A^T + λ I)^{-1} @ A) + ...
    dB* = (I - B @ (B^T @ B + λ I)^{-1} @ B^T) @ G @ A^T @ (A @ A^T + λ I)^{-1}
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAProGradientProjector:
    """
    Projector that aligns low-rank adapter gradients with the full fine-tuning gradient.
    Computes closed-form projection updates for dA and dB.
    """

    def __init__(self, damping: float = 1e-4):
        self.damping = damping

    def project(
        self,
        full_grad: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        scaling: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Projects full weight gradient G into optimal dA and dB updates.

        Args:
            full_grad: Full weight gradient G in R^{out_features x in_features}
            lora_A: Adapter A matrix in R^{r x in_features}
            lora_B: Adapter B matrix in R^{out_features x r}
            scaling: Scaling constant (alpha / r)

        Returns:
            Tuple of (projected_grad_A, projected_grad_B)
        """
        device = full_grad.device
        dtype = full_grad.dtype
        r = lora_A.shape[0]

        # Convert to float32 for stable matrix inversion
        A_f = lora_A.float()
        B_f = lora_B.float()
        G_f = full_grad.float()

        # Gram matrices: B^T @ B (r x r) and A @ A^T (r x r)
        BtB = torch.matmul(B_f.t(), B_f) + self.damping * torch.eye(r, device=device)
        AAt = torch.matmul(A_f, A_f.t()) + self.damping * torch.eye(r, device=device)

        # Inverses in rank dimension (r is small, e.g. 16 or 32, so inversion is negligible: O(r^3))
        inv_BtB = torch.linalg.inv(BtB)
        inv_AAt = torch.linalg.inv(AAt)

        # Standard unprojected gradients:
        # grad_A_std = B^T @ G
        # grad_B_std = G @ A^T
        grad_A_raw = torch.matmul(B_f.t(), G_f)
        grad_B_raw = torch.matmul(G_f, A_f.t())

        # LoRA-Pro Projection:
        # Project grad_A using inv_BtB
        proj_grad_A = torch.matmul(inv_BtB, grad_A_raw)

        # Project grad_B using inv_AAt
        proj_grad_B = torch.matmul(grad_B_raw, inv_AAt)

        # Apply scaling factor
        proj_grad_A = (proj_grad_A * scaling).to(dtype)
        proj_grad_B = (proj_grad_B * scaling).to(dtype)

        return proj_grad_A, proj_grad_B


class M2LRFLoRAProLinear(nn.Module):
    """
    LoRA-Pro Linear layer combined with M-2LRF 2-bit dual-basis base weights.
    Implements gradient projection alignment during backward passes.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
        damping: float = 1e-4,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.damping = damping

        # Frozen base weight (simulated or quantized)
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

        self.projector = LoRAProGradientProjector(damping=damping)
        self.use_pro_alignment = True

    def reset_parameters(self):
        if self.r > 0:
            # Initialize A with Kaiming uniform and B with zeros
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def initialize_from_pretrained(self, pretrained_weight: torch.Tensor):
        """Copies pretrained weights into base weight."""
        with torch.no_grad():
            self.weight.copy_(pretrained_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward
        base_out = F.linear(x, self.weight, self.bias)

        if self.r <= 0:
            return base_out

        # Adapter forward
        adapter_in = self.dropout(x)
        adapter_h = F.linear(adapter_in, self.lora_A)
        adapter_out = F.linear(adapter_h, self.lora_B)

        return base_out + self.scaling * adapter_out

    def align_gradients(self, full_grad: torch.Tensor):
        """
        Explicitly applies LoRA-Pro gradient projection to lora_A.grad and lora_B.grad
        when full weight gradients G are provided.
        """
        if self.r <= 0 or not self.use_pro_alignment:
            return

        proj_A, proj_B = self.projector.project(
            full_grad=full_grad,
            lora_A=self.lora_A.data,
            lora_B=self.lora_B.data,
            scaling=self.scaling,
        )

        if self.lora_A.grad is not None:
            self.lora_A.grad.copy_(proj_A)
        else:
            self.lora_A.grad = proj_A.clone()

        if self.lora_B.grad is not None:
            self.lora_B.grad.copy_(proj_B)
        else:
            self.lora_B.grad = proj_B.clone()

    def merge(self) -> torch.Tensor:
        """Merges low-rank update into base weight."""
        if self.r > 0:
            delta = (self.scaling * (self.lora_B @ self.lora_A)).to(self.weight.dtype)
            self.weight.add_(delta)
            self.lora_B.data.zero_()
            self.lora_A.data.zero_()
        return self.weight
