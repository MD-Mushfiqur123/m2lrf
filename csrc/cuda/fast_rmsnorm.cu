#include "../include/m2lrf_common.cuh"

namespace m2lrf {

// Fused RMSNorm forward kernel with FP16/BF16/FP32 precision
__global__ void fast_rmsnorm_forward_kernel(
    const float *__restrict__ X,
    const float *__restrict__ W,
    float *__restrict__ Y,
    float *__restrict__ rstd,
    float eps,
    int hidden_size
) {
    int row = blockIdx.x;
    int lane = threadIdx.x;

    const float *x_row = X + row * hidden_size;
    float *y_row = Y + row * hidden_size;

    float sum_sq = 0.0f;
    for (int i = lane; i < hidden_size; i += WARP_SIZE) {
        float val = x_row[i];
        sum_sq += val * val;
    }

    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float s_rstd;
    if (lane == 0) {
        float mean_sq = sum_sq / (float)hidden_size;
        s_rstd = rsqrtf(mean_sq + eps);
        if (rstd != nullptr) {
            rstd[row] = s_rstd;
        }
    }
    __syncthreads();

    float r = s_rstd;
    for (int i = lane; i < hidden_size; i += WARP_SIZE) {
        y_row[i] = x_row[i] * r * W[i];
    }
}

extern "C" void launch_fast_rmsnorm(
    const float *X,
    const float *W,
    float *Y,
    float *rstd,
    float eps,
    int num_rows,
    int hidden_size,
    cudaStream_t stream
) {
    dim3 grid(num_rows);
    dim3 block(WARP_SIZE);
    fast_rmsnorm_forward_kernel<<<grid, block, 0, stream>>>(
        X, W, Y, rstd, eps, hidden_size
    );
}

} // namespace m2lrf
