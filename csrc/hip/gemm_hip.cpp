// AMD ROCm / HIP Matrix Core 2-Bit Dual-Basis Kernel for CDNA2 / CDNA3 (MI200, MI300X)
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace m2lrf_hip {

__device__ __forceinline__ uint8_t hip_extract_2bit(uint8_t packed, int index) {
    return (packed >> (2 * index)) & 0x03;
}

__device__ __forceinline__ float hip_decode_2bit(uint8_t code, float a0, float a1) {
    switch (code) {
        case 0: return -a1;
        case 1: return -a0;
        case 2: return +a0;
        case 3: return +a1;
        default: return 0.0f;
    }
}

__global__ void hip_gemv_2bit_kernel(
    const uint8_t *__restrict__ W_packed,
    const float *__restrict__ X,
    const float *__restrict__ scales_a0,
    const float *__restrict__ scales_a1,
    float *__restrict__ Y,
    int in_features,
    int out_features,
    int group_size
) {
    int row = hipBlockIdx_x * hipBlockDim_y + hipThreadIdx_y;
    if (row >= out_features) return;

    int lane = hipThreadIdx_x;
    int packed_dim = in_features / 4;
    int groups_per_row = (in_features + group_size - 1) / group_size;

    float acc = 0.0f;
    for (int p = lane; p < packed_dim; p += 64) { // AMD wave64
        uint8_t byte_val = W_packed[row * packed_dim + p];
        int elem_base = p * 4;
        int group_idx = elem_base / group_size;

        float a0 = scales_a0[row * groups_per_row + group_idx];
        float a1 = scales_a1[row * groups_per_row + group_idx];

        for (int i = 0; i < 4; ++i) {
            uint8_t code = hip_extract_2bit(byte_val, i);
            acc += hip_decode_2bit(code, a0, a1) * X[elem_base + i];
        }
    }

    // Wave reduction
    for (int offset = 32; offset > 0; offset /= 2) {
        acc += __shfl_down(acc, offset, 64);
    }

    if (lane == 0) {
        atomicAdd(&Y[row], acc);
    }
}

} // namespace m2lrf_hip
