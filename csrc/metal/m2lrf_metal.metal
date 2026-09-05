#include <metal_stdlib>
using namespace metal;

// 2-Bit Dual-Basis Metal Shading Language (MSL) Compute Kernel for Apple Silicon
inline float decode_2bit_metal(uint8_t code, float a0, float a1) {
    switch (code) {
        case 0: return -a1;
        case 1: return -a0;
        case 2: return +a0;
        case 3: return +a1;
        default: return 0.0f;
    }
}

kernel void m2lrf_gemv_2bit_metal(
    device const uint8_t *W_packed [[buffer(0)]],
    device const float *X [[buffer(1)]],
    device const float *scales_a0 [[buffer(2)]],
    device const float *scales_a1 [[buffer(3)]],
    device float *Y [[buffer(4)]],
    constant uint &in_features [[buffer(5)]],
    constant uint &out_features [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    uint row = gid.y;
    if (row >= out_features) return;

    uint packed_dim = in_features / 4;
    uint groups_per_row = (in_features + group_size - 1) / group_size;

    float acc = 0.0f;
    for (uint p = simd_lane_id; p < packed_dim; p += 32) {
        uint8_t byte_val = W_packed[row * packed_dim + p];
        uint elem_base = p * 4;
        uint group_idx = elem_base / group_size;

        float a0 = scales_a0[row * groups_per_row + group_idx];
        float a1 = scales_a1[row * groups_per_row + group_idx];

        for (uint i = 0; i < 4; ++i) {
            uint8_t code = (byte_val >> (2 * i)) & 0x03;
            acc += decode_2bit_metal(code, a0, a1) * X[elem_base + i];
        }
    }

    // SIMD group reduction
    acc = simd_sum(acc);

    if (simd_lane_id == 0) {
        Y[row] += acc;
    }
}
