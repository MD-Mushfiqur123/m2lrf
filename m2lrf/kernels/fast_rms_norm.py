"""
M-2LRF Fast RMSNorm Kernel (Unsloth-Inspired)
==============================================
Fused RMSNorm forward and backward Triton kernel eliminating memory allocation
for intermediate mean-square and reciprocal square-root activations.
"""

import math
from typing import Optional
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _rms_norm_fwd_kernel(
        x_ptr,
        y_ptr,
        w_ptr,
        r_ptr,
        stride_x_row,
        stride_x_col,
        stride_y_row,
        stride_y_col,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr
    ):
        row_idx = tl.program_id(0)
        row_start_x = x_ptr + row_idx * stride_x_row
        row_start_y = y_ptr + row_idx * stride_y_row

        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        x = tl.load(row_start_x + cols * stride_x_col, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)

        # Compute variance: mean(x^2)
        variance = tl.sum(x * x, axis=0) / n_cols
        rsqrt_val = 1.0 / tl.sqrt(variance + eps)
        tl.store(r_ptr + row_idx, rsqrt_val)

        # Normalize and scale
        y = (x * rsqrt_val) * w
        tl.store(row_start_y + cols * stride_y_col, y, mask=mask)

    @triton.jit
    def _rms_norm_bwd_kernel(
        dy_ptr,
        x_ptr,
        w_ptr,
        r_ptr,
        dx_ptr,
        dw_ptr,
        stride_dy_row,
        stride_dy_col,
        stride_x_row,
        stride_x_col,
        stride_dx_row,
        stride_dx_col,
        n_rows,
        n_cols,
        BLOCK_SIZE: tl.constexpr
    ):
        row_idx = tl.program_id(0)
        row_start_dy = dy_ptr + row_idx * stride_dy_row
        row_start_x = x_ptr + row_idx * stride_x_row
        row_start_dx = dx_ptr + row_idx * stride_dx_row

        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        dy = tl.load(row_start_dy + cols * stride_dy_col, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(row_start_x + cols * stride_x_col, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        r = tl.load(r_ptr + row_idx).to(tl.float32)

        # Compute inner product dy * (x * w)
        dy_w = dy * w
        sum_dy_w_x = tl.sum(dy_w * x, axis=0)
        
        # dL/dx = r * (dy * w - (x * r^2 / n_cols) * sum(dy * w * x))
        dx = r * (dy_w - (x * (r * r / n_cols)) * sum_dy_w_x)
        tl.store(row_start_dx + cols * stride_dx_col, dx, mask=mask)


class FastRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
        orig_shape = x.shape
        x_2d = x.view(-1, orig_shape[-1]).contiguous()
        n_rows, n_cols = x_2d.shape

        if HAS_TRITON and x.is_cuda:
            BLOCK_SIZE = min(8192, triton.next_power_of_2(n_cols))
            y_2d = torch.empty_like(x_2d)
            r_vec = torch.empty(n_rows, device=x.device, dtype=torch.float32)

            _rms_norm_fwd_kernel[(n_rows,)](
                x_2d, y_2d, weight, r_vec,
                x_2d.stride(0), x_2d.stride(1),
                y_2d.stride(0), y_2d.stride(1),
                n_cols, eps,
                BLOCK_SIZE=BLOCK_SIZE
            )
            ctx.save_for_backward(x_2d, weight, r_vec)
            ctx.orig_shape = orig_shape
            return y_2d.view(orig_shape)
        else:
            variance = x_2d.pow(2).mean(dim=-1, keepdim=True)
            rsqrt_val = torch.rsqrt(variance + eps)
            norm_x = x_2d * rsqrt_val
            y_2d = norm_x * weight
            ctx.save_for_backward(norm_x, weight, rsqrt_val)
            ctx.orig_shape = orig_shape
            return y_2d.view(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        orig_shape = ctx.orig_shape
        dy_2d = grad_output.view(-1, orig_shape[-1]).contiguous()
        saved = ctx.saved_tensors

        if HAS_TRITON and dy_2d.is_cuda and len(saved) == 3 and saved[2].ndim == 1:
            x_2d, weight, r_vec = saved
            n_rows, n_cols = x_2d.shape
            BLOCK_SIZE = min(8192, triton.next_power_of_2(n_cols))
            dx_2d = torch.empty_like(x_2d)

            _rms_norm_bwd_kernel[(n_rows,)](
                dy_2d, x_2d, weight, r_vec, dx_2d, None,
                dy_2d.stride(0), dy_2d.stride(1),
                x_2d.stride(0), x_2d.stride(1),
                dx_2d.stride(0), dx_2d.stride(1),
                n_rows, n_cols,
                BLOCK_SIZE=BLOCK_SIZE
            )
            dw = (dy_2d * (x_2d * r_vec.unsqueeze(1))).sum(dim=0).to(weight.dtype)
            return dx_2d.view(orig_shape), dw, None
        else:
            norm_x, weight, rsqrt_val = saved
            dw = (dy_2d * norm_x).sum(dim=0).to(weight.dtype)
            dy_w = dy_2d * weight
            dx_2d = rsqrt_val * (dy_w - norm_x * (dy_w * norm_x).mean(dim=-1, keepdim=True))
            return dx_2d.view(orig_shape), dw, None


class FastRMSNorm(nn.Module):
    """
    Fused drop-in replacement for LlamaRMSNorm and Qwen2RMSNorm.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return FastRMSNormFunction.apply(x, self.weight, self.eps)

    def extra_repr(self):
        return f"{self.weight.shape[0]}, eps={self.eps}"
