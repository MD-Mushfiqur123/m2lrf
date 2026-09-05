#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdint.h>

#define WARP_SIZE 32
#define FULL_WARP_MASK 0xffffffff

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

namespace m2lrf {

// Lloyd-Max Canonical Centroids for Standard Gaussian Distribution
__constant__ float DUAL_BASIS_A0 = 0.528622f;
__constant__ float DUAL_BASIS_A1 = 1.603332f;
__constant__ float DUAL_BASIS_TAU = 1.065977f;

// Bit-unpacking helper: extracts 2-bit code from packed uint8 byte
__device__ __forceinline__ uint8_t extract_2bit(uint8_t packed, int index) {
    return (packed >> (2 * index)) & 0x03;
}

// Maps 2-bit code [0, 1, 2, 3] to ternary dual-basis value:
// 0 -> -a1, 1 -> -a0, 2 -> +a0, 3 -> +a1
__device__ __forceinline__ float decode_2bit_scalar(uint8_t code, float a0, float a1) {
    switch (code) {
        case 0: return -a1;
        case 1: return -a0;
        case 2: return +a0;
        case 3: return +a1;
        default: return 0.0f;
    }
}

// Warp reduction helper for float summation
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
    }
    return val;
}

// Warp reduction helper for float maximum
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(FULL_WARP_MASK, val, offset));
    }
    return val;
}

} // namespace m2lrf
