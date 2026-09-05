#include "../include/m2lrf_common.cuh"
#include "../include/m2lrf_hadamard.cuh"

namespace m2lrf {

// 2-Bit Dual-Basis GEMV Kernel: Y = W_2bit * X + LoRA_B * (LoRA_A * X)
// Designed for single-token autoregressive decoding
__global__ void gemv_2bit_kernel(
    const uint8_t *__restrict__ W_packed,  // [out_features, in_features / 4]
    const float *__restrict__ X,           // [in_features]
    const float *__restrict__ scales_a0,   // [out_features, num_groups]
    const float *__restrict__ scales_a1,   // [out_features, num_groups]
    float *__restrict__ Y,                 // [out_features]
    int in_features,
    int out_features,
    int group_size
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= out_features) return;

    int lane = threadIdx.x;
    int packed_dim = in_features / 4;
    int groups_per_row = (in_features + group_size - 1) / group_size;

    float acc = 0.0f;

    // Each warp processes elements along the row
    for (int p = lane; p < packed_dim; p += WARP_SIZE) {
        uint8_t byte_val = W_packed[row * packed_dim + p];
        int elem_base = p * 4;
        int group_idx = elem_base / group_size;

        float a0 = scales_a0[row * groups_per_row + group_idx];
        float a1 = scales_a1[row * groups_per_row + group_idx];

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint8_t code = extract_2bit(byte_val, i);
            float weight = decode_2bit_scalar(code, a0, a1);
            acc += weight * X[elem_base + i];
        }
    }

    // Warp-level reduction
    acc = warp_reduce_sum(acc);

    if (lane == 0) {
        atomicAdd(&Y[row], acc);
    }
}

// Host launcher function
extern "C" void launch_gemv_2bit(
    const uint8_t *W_packed,
    const float *X,
    const float *scales_a0,
    const float *scales_a1,
    float *Y,
    int in_features,
    int out_features,
    int group_size,
    cudaStream_t stream
) {
    dim3 block(WARP_SIZE, 8);
    dim3 grid((out_features + block.y - 1) / block.y);
    gemv_2bit_kernel<<<grid, block, 0, stream>>>(
        W_packed, X, scales_a0, scales_a1, Y, in_features, out_features, group_size
    );
}

} // namespace m2lrf
