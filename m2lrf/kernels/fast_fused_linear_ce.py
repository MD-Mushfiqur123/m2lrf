"""
M-2LRF Fused Linear + Cross-Entropy Kernel (Liger-Kernel-Inspired)
===================================================================
Fuses the final lm_head linear projection with cross-entropy loss computation.
Completely bypasses materializing the [Batch, SeqLen, VocabSize] logits tensor in VRAM!
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        chunk_size: int = 1024
    ) -> torch.Tensor:
        # x: [N, hidden_dim]
        # weight: [vocab_size, hidden_dim]
        # targets: [N]
        orig_shape = x.shape
        x_2d = x.view(-1, orig_shape[-1])
        targets_flat = targets.view(-1)
        N, H = x_2d.shape
        V, _ = weight.shape

        loss_total = torch.zeros(N, device=x.device, dtype=torch.float32)
        valid_mask = (targets_flat != ignore_index)

        # Chunked forward
        for i in range(0, N, chunk_size):
            chunk_x = x_2d[i:i+chunk_size]
            chunk_targets = targets_flat[i:i+chunk_size]
            chunk_logits = F.linear(chunk_x, weight, bias)
            chunk_loss = F.cross_entropy(chunk_logits, chunk_targets, ignore_index=ignore_index, reduction='none')
            loss_total[i:i+chunk_size] = chunk_loss

        ctx.save_for_backward(x_2d, weight, targets_flat, bias)
        ctx.ignore_index = ignore_index
        ctx.chunk_size = chunk_size
        ctx.orig_shape = orig_shape

        return loss_total.view(*orig_shape[:-1])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_2d, weight, targets_flat, bias = ctx.saved_tensors
        ignore_index = ctx.ignore_index
        chunk_size = ctx.chunk_size
        orig_shape = ctx.orig_shape

        N, H = x_2d.shape
        V, _ = weight.shape
        grad_flat = grad_output.view(-1, 1)

        dx = torch.zeros_like(x_2d)
        dweight = torch.zeros_like(weight)
        dbias = torch.zeros_like(bias) if bias is not None else None

        valid_mask = (targets_flat != ignore_index)

        for i in range(0, N, chunk_size):
            chunk_x = x_2d[i:i+chunk_size]
            chunk_targets = targets_flat[i:i+chunk_size]
            chunk_grad = grad_flat[i:i+chunk_size]
            chunk_valid = valid_mask[i:i+chunk_size]

            chunk_logits = F.linear(chunk_x, weight, bias)
            probs = F.softmax(chunk_logits, dim=-1)

            target_one_hot = F.one_hot(chunk_targets.clamp(min=0), num_classes=V).to(probs.dtype)
            dlogits = (probs - target_one_hot) * chunk_grad
            dlogits[~chunk_valid] = 0.0

            # Accumulate gradients
            dx[i:i+chunk_size] = dlogits.matmul(weight)
            dweight.add_(dlogits.t().matmul(chunk_x))
            if dbias is not None:
                dbias.add_(dlogits.sum(dim=0))

        return dx.view(orig_shape), dweight, None, dbias, None, None


def fused_linear_cross_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Fused Linear Projection + Cross Entropy.
    """
    loss = FusedLinearCrossEntropyFunction.apply(x, weight, targets, bias, ignore_index)
    if reduction == "none":
        return loss
    valid_count = (targets != ignore_index).sum().clamp(min=1)
    if reduction == "mean":
        return loss.sum() / valid_count
    elif reduction == "sum":
        return loss.sum()
    return loss
