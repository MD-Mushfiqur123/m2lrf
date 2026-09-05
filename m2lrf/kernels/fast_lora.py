"""
M-2LRF Fast LoRA Kernel (Unsloth-Inspired)
===========================================
Fused low-rank adapter linear forward and backward kernel.
Optimizes intermediate activation lifetimes for LoRA branches h = (alpha / rank) * (X @ A^T) @ B^T.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FastLoRAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor, scaling: float):
        # x: [batch * seq, in_features]
        # lora_A: [rank, in_features]
        # lora_B: [out_features, rank]
        ctx.save_for_backward(x, lora_A, lora_B)
        ctx.scaling = scaling

        # (x @ A^T) @ B^T
        r_intermediate = F.linear(x, lora_A)  # [N, rank]
        out = F.linear(r_intermediate, lora_B) * scaling  # [N, out_features]
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, lora_A, lora_B = ctx.saved_tensors
        scaling = ctx.scaling

        # grad_output: [N, out_features]
        scaled_grad = grad_output * scaling

        # d(lora_B) = scaled_grad^T @ (x @ A^T)
        r_intermediate = F.linear(x, lora_A)  # [N, rank]
        d_lora_B = scaled_grad.t().matmul(r_intermediate)  # [out_features, rank]

        # d(intermediate) = scaled_grad @ lora_B -> [N, rank]
        d_intermediate = scaled_grad.matmul(lora_B)

        # d(lora_A) = d_intermediate^T @ x -> [rank, in_features]
        d_lora_A = d_intermediate.t().matmul(x)

        # d(x) = d_intermediate @ lora_A -> [N, in_features]
        dx = d_intermediate.matmul(lora_A)

        return dx, d_lora_A, d_lora_B, None


def fast_lora_forward(
    x: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float
) -> torch.Tensor:
    """
    Fused LoRA branch computation with minimal memory retention.
    """
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    out_2d = FastLoRAFunction.apply(x_2d, lora_A, lora_B, scaling)
    return out_2d.view(*orig_shape[:-1], lora_B.shape[0])
