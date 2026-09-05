"""
M-2LRF Fast SwiGLU Kernel (Unsloth-Inspired)
==============================================
Fused SwiGLU (SiLU(gate) * up) forward and backward kernel.
Halves activation memory cached during transformer MLP forward pass.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _swiglu_fwd_kernel(
        gate_ptr,
        up_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # silu(gate) = gate / (1 + exp(-gate))
        silu_gate = gate * tl.sigmoid(gate)
        out = silu_gate * up
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _swiglu_bwd_kernel(
        dout_ptr,
        gate_ptr,
        up_ptr,
        dgate_ptr,
        dup_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        dout = tl.load(dout_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        sig_gate = tl.sigmoid(gate)
        silu_gate = gate * sig_gate

        # dup = dout * silu(gate)
        dup = dout * silu_gate
        tl.store(dup_ptr + offsets, dup, mask=mask)

        # dgate = dout * up * (sig + gate * sig * (1 - sig))
        d_silu = sig_gate + gate * sig_gate * (1.0 - sig_gate)
        dgate = dout * up * d_silu
        tl.store(dgate_ptr + offsets, dgate, mask=mask)


class FastSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor):
        n_elements = gate.numel()
        ctx.save_for_backward(gate, up)

        if HAS_TRITON and gate.is_cuda and up.is_cuda:
            out = torch.empty_like(gate)
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            _swiglu_fwd_kernel[grid](gate, up, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
            return out
        else:
            return F.silu(gate) * up

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        gate, up = ctx.saved_tensors
        n_elements = gate.numel()

        if HAS_TRITON and dout.is_cuda and gate.is_cuda:
            dgate = torch.empty_like(gate)
            dup = torch.empty_like(up)
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            _swiglu_bwd_kernel[grid](dout, gate, up, dgate, dup, n_elements, BLOCK_SIZE=BLOCK_SIZE)
            return dgate, dup
        else:
            sig = torch.sigmoid(gate)
            silu = gate * sig
            dup = dout * silu
            dgate = dout * up * (sig + silu * (1.0 - sig))
            return dgate, dup


def fast_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Fused drop-in replacement for F.silu(gate) * up.
    """
    return FastSwiGLUFunction.apply(gate, up)
