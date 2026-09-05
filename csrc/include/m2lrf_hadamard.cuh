#pragma once

#include "m2lrf_common.cuh"

namespace m2lrf {

// In-warp Fast Walsh-Hadamard Transform (FWHT) for vectors of length 32
__device__ __forceinline__ void fwht_warp_32(float &val) {
    #pragma unroll
    for (int stride = 1; stride < WARP_SIZE; stride *= 2) {
        float other = __shfl_xor_sync(FULL_WARP_MASK, val, stride);
        int lane_id = threadIdx.x % WARP_SIZE;
        if ((lane_id & stride) == 0) {
            val = val + other;
        } else {
            val = other - val;
        }
    }
    // Normalization factor 1 / sqrt(32)
    val *= 0.17677669529663687f;
}

// In-shared-memory Block Fast Walsh-Hadamard Transform for block sizes up to 1024
template <int BLOCK_SIZE>
__device__ void fwht_shared_block(float *smem, int tid) {
    #pragma unroll
    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        __syncthreads();
        int chunk = tid / stride;
        int i = (chunk * 2) * stride + (tid % stride);
        if (i + stride < BLOCK_SIZE) {
            float u = smem[i];
            float v = smem[i + stride];
            smem[i] = u + v;
            smem[i + stride] = u - v;
        }
    }
    __syncthreads();
    // Normalize by 1 / sqrt(BLOCK_SIZE)
    float norm = 1.0f / sqrtf((float)BLOCK_SIZE);
    smem[tid] *= norm;
}

} // namespace m2lrf
