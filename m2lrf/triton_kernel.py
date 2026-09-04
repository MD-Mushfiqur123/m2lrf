"""
M-2LRF Native Triton Dequantization & GEMM Kernel
=================================================
Features:
  1. High-throughput 2-bit bit-unpacking via fast bitwise GPU intrinsics.
  2. In-SRAM Fused Dequant + Matmul (Zero global memory write for FP16 weights).
  3. Seamless vectorized PyTorch CPU/CUDA fallback when Triton is unavailable.
"""

import math
from typing import Tuple, Optional, Union
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
    def _fused_2bit_dequant_gemm_kernel(
        # Pointers
        x_ptr,           # Input activation X: [M, K]
        w_packed_ptr,    # Packed 2-bit weights: [N, K // 4] (uint8)
        a0_ptr,          # Alpha_0 scale per row: [N, 1] or [N]
        a1_ptr,          # Alpha_1 scale per row: [N, 1] or [N]
        out_ptr,         # Output matrix: [M, N]
        # Dimensions
        M, N, K,
        # Strides
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_om, stride_on,
        # Meta-parameters
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Fused Triton kernel for 2-bit in-SRAM dequantization and GEMM.
        Performs in-SRAM bit-unpacking and accumulation directly into registers:
            Y = X @ W_dequant^T
        where W_dequant has shape [N, K] reconstructed on-the-fly from uint8 packed bytes [N, K // 4].
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Accumulator in FP32
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Load per-row scale factors alpha_0 and alpha_1 for this N-block
        a0 = tl.load(a0_ptr + offs_n[:, None], mask=offs_n[:, None] < N, other=0.0)
        a1 = tl.load(a1_ptr + offs_n[:, None], mask=offs_n[:, None] < N, other=0.0)

        SUB_K: tl.constexpr = BLOCK_K // 4

        for k_iter in range(0, tl.cdiv(K, BLOCK_K)):
            k_base = k_iter * BLOCK_K
            k_sub_base = k_iter * SUB_K
            sub_idx = tl.arange(0, SUB_K)

            # 4 interleaved column offsets in activation matrix X
            k0 = k_base + sub_idx * 4 + 0
            k1 = k_base + sub_idx * 4 + 1
            k2 = k_base + sub_idx * 4 + 2
            k3 = k_base + sub_idx * 4 + 3

            # Load 4 sub-tiles of X, each of shape [BLOCK_M, SUB_K]
            x0 = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + k0[None, :] * stride_xk,
                mask=(offs_m[:, None] < M) & (k0[None, :] < K),
                other=0.0
            )
            x1 = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + k1[None, :] * stride_xk,
                mask=(offs_m[:, None] < M) & (k1[None, :] < K),
                other=0.0
            )
            x2 = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + k2[None, :] * stride_xk,
                mask=(offs_m[:, None] < M) & (k2[None, :] < K),
                other=0.0
            )
            x3 = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + k3[None, :] * stride_xk,
                mask=(offs_m[:, None] < M) & (k3[None, :] < K),
                other=0.0
            )

            # Load packed weight bytes of shape [BLOCK_N, SUB_K]
            k_packed = k_sub_base + sub_idx
            w_mask = (offs_n[:, None] < N) & (k_packed[None, :] < (K // 4))
            packed_bytes = tl.load(
                w_packed_ptr + offs_n[:, None] * stride_wn + k_packed[None, :] * stride_wk,
                mask=w_mask,
                other=0
            )

            # Bit-unpack 4 2-bit codes per byte
            c0 = (packed_bytes >> 0) & 0x03
            c1 = (packed_bytes >> 2) & 0x03
            c2 = (packed_bytes >> 4) & 0x03
            c3 = (packed_bytes >> 6) & 0x03

            # Dual-basis dequantization mapping in SRAM:
            # code 0: -alpha1
            # code 1: -alpha0
            # code 2: +alpha0
            # code 3: +alpha1
            v0 = tl.where(c0 == 0, -a1, tl.where(c0 == 1, -a0, tl.where(c0 == 2, a0, a1))).to(tl.float16)
            v1 = tl.where(c1 == 0, -a1, tl.where(c1 == 1, -a0, tl.where(c1 == 2, a0, a1))).to(tl.float16)
            v2 = tl.where(c2 == 0, -a1, tl.where(c2 == 1, -a0, tl.where(c2 == 2, a0, a1))).to(tl.float16)
            v3 = tl.where(c3 == 0, -a1, tl.where(c3 == 1, -a0, tl.where(c3 == 2, a0, a1))).to(tl.float16)

            # Fused Tensor Core GEMM accumulation into registers
            acc += tl.dot(x0.to(tl.float16), tl.trans(v0))
            acc += tl.dot(x1.to(tl.float16), tl.trans(v1))
            acc += tl.dot(x2.to(tl.float16), tl.trans(v2))
            acc += tl.dot(x3.to(tl.float16), tl.trans(v3))

        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc.to(tl.float16), mask=out_mask)


def m2lrf_matmul_fallback(
    x: torch.Tensor,
    packed_weights: torch.Tensor,
    a0: torch.Tensor,
    a1: torch.Tensor,
    orig_shape: Tuple[int, ...]
) -> torch.Tensor:
    """
    Vectorized high-performance PyTorch fallback for CPU / CUDA environments without Triton.
    """
    c0 = (packed_weights >> 0) & 0x03
    c1 = (packed_weights >> 2) & 0x03
    c2 = (packed_weights >> 4) & 0x03
    c3 = (packed_weights >> 6) & 0x03

    codes = torch.stack([c0, c1, c2, c3], dim=-1).flatten(start_dim=-2)
    codes = codes[..., :orig_shape[-1]]

    w_dequant = torch.zeros(orig_shape, dtype=torch.float16, device=packed_weights.device)
    w_dequant = torch.where(codes == 0, -a1, w_dequant)
    w_dequant = torch.where(codes == 1, -a0, w_dequant)
    w_dequant = torch.where(codes == 2, a0, w_dequant)
    w_dequant = torch.where(codes == 3, a1, w_dequant)

    return F.linear(x, w_dequant.to(x.dtype))


def m2lrf_triton_matmul(
    x: torch.Tensor,
    packed_weights: torch.Tensor,
    a0: torch.Tensor,
    a1: torch.Tensor,
    orig_shape: Tuple[int, ...]
) -> torch.Tensor:
    """
    High-level dispatch for M-2LRF Matmul.
    Uses fused Triton kernel if CUDA and Triton are available; otherwise routes to vectorized PyTorch fallback.
    """
    if not (HAS_TRITON and x.is_cuda and packed_weights.is_cuda):
        return m2lrf_matmul_fallback(x, packed_weights, a0, a1, orig_shape)

    # Flatten leading batch dimensions if necessary
    orig_x_shape = x.shape
    x_2d = x.view(-1, orig_x_shape[-1]).contiguous()
    M, K = x_2d.shape
    N = orig_shape[0]

    out = torch.empty((M, N), device=x.device, dtype=torch.float16)

    BLOCK_M = 32 if M <= 32 else 64
    BLOCK_N = 64
    BLOCK_K = 64  # SUB_K = 16, suitable for Tensor Core dot product

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N)
    )

    _fused_2bit_dequant_gemm_kernel[grid](
        x_2d,
        packed_weights,
        a0.contiguous(),
        a1.contiguous(),
        out,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        packed_weights.stride(0), packed_weights.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )

    return out.view(*orig_x_shape[:-1], N).to(x.dtype)


__all__ = [
    "HAS_TRITON",
    "_fused_2bit_dequant_gemm_kernel",
    "m2lrf_triton_matmul",
    "m2lrf_matmul_fallback"
]
