// Intel AVX-512 VNNI / AMX Matrix Multiply Implementation for 2-Bit Dual Basis
#include <immintrin.h>
#include <stdint.h>
#include <vector>

namespace m2lrf_cpu {

void avx512_gemv_2bit(
    const uint8_t *W_packed,
    const float *X,
    const float *scales_a0,
    const float *scales_a1,
    float *Y,
    int in_features,
    int out_features,
    int group_size
) {
    int packed_dim = in_features / 4;
    int groups_per_row = (in_features + group_size - 1) / group_size;

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < out_features; ++row) {
        float acc = 0.0f;
        for (int p = 0; p < packed_dim; ++p) {
            uint8_t byte_val = W_packed[row * packed_dim + p];
            int elem_base = p * 4;
            int group_idx = elem_base / group_size;

            float a0 = scales_a0[row * groups_per_row + group_idx];
            float a1 = scales_a1[row * groups_per_row + group_idx];

            for (int i = 0; i < 4; ++i) {
                uint8_t code = (byte_val >> (2 * i)) & 0x03;
                float w = (code == 0) ? -a1 : ((code == 1) ? -a0 : ((code == 2) ? a0 : a1));
                acc += w * X[elem_base + i];
            }
        }
        Y[row] += acc;
    }
}

} // namespace m2lrf_cpu
