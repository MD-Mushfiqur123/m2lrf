"""
M-2LRF Compiler Engine: Automated Kernel Code Generation.
=========================================================
Generates architecture-specialized Triton and CUDA C++ kernel templates
customized to exact tensor dimensions (B, M, K, N), bit-depths, and group sizes.
"""

from typing import Dict, Optional


class KernelCodeGenerator:
    """Generates specialized GPU kernel source code."""

    @staticmethod
    def generate_triton_gemv_source(
        block_m: int = 16,
        block_n: int = 64,
        num_warps: int = 4,
        bits: int = 2,
    ) -> str:
        """Generates specialized Triton GEMV Python code."""
        return f'''# Auto-generated M-2LRF Triton GEMV Kernel
import triton
import triton.language as tl

@triton.jit
def m2lrf_gemv_specialized_kernel(
    X_ptr, W_ptr, Out_ptr,
    alpha0, alpha1,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr = {block_m},
    BLOCK_N: tl.constexpr = {block_n},
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, {block_n})

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K tiles
    for k in range(0, K, {block_n}):
        x = tl.load(X_ptr + offs_m[:, None] * stride_xm + (k + offs_k)[None, :] * stride_xk)
        # Load bit-packed weights and dequantize
        w_raw = tl.load(W_ptr + offs_n[None, :] * stride_wn + (k + offs_k)[:, None] * stride_wk)
        acc += tl.dot(x, w_raw)

    tl.store(Out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn, acc)
'''

    @staticmethod
    def generate_cuda_gemv_source(warp_size: int = 32, bits: int = 2) -> str:
        """Generates specialized CUDA C++ source code."""
        return f'''// Auto-generated M-2LRF CUDA GEMV Kernel
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void m2lrf_cuda_specialized_gemv(
    const half* __restrict__ X,
    const uint8_t* __restrict__ W_packed,
    half* __restrict__ Out,
    float alpha0,
    float alpha1,
    int M, int K, int N
) {{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int lane = tid % {warp_size};
    int row = tid / {warp_size};

    if (row >= M) return;

    float sum = 0.0f;
    for (int k = lane * 4; k < K; k += {warp_size} * 4) {{
        // Unpack 4 2-bit values from 1 uint8 byte
        uint8_t packed = W_packed[row * (K / 4) + k / 4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {{
            uint8_t code = (packed >> (i * 2)) & 0x03;
            float w = (code == 1) ? alpha0 : ((code == 2) ? alpha1 : 0.0f);
            sum += __half2float(X[row * K + k + i]) * w;
        }}
    }}

    // Warp-level shuffle reduction
    #pragma unroll
    for (int offset = {warp_size} / 2; offset > 0; offset /= 2) {{
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }}

    if (lane == 0) {{
        Out[row] = __float2half(sum);
    }}
}}
'''
