#include "../include/m2lrf_common.cuh"

namespace m2lrf {

// Fused in-place Rotary Position Embeddings (RoPE) kernel
__global__ void fast_rope_kernel(
    float *__restrict__ Q,         // [batch, seq_len, num_heads, head_dim]
    float *__restrict__ K,         // [batch, seq_len, num_kv_heads, head_dim]
    const float *__restrict__ cos, // [seq_len, head_dim / 2]
    const float *__restrict__ sin, // [seq_len, head_dim / 2]
    int batch_size,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    int half_dim = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_q_pairs = batch_size * seq_len * num_heads * half_dim;

    if (idx < total_q_pairs) {
        int d = idx % half_dim;
        int temp = idx / half_dim;
        int h = temp % num_heads;
        temp = temp / num_heads;
        int s = temp % seq_len;
        int b = temp / seq_len;

        float c = cos[s * half_dim + d];
        float sn = sin[s * half_dim + d];

        int q_base = ((b * seq_len + s) * num_heads + h) * head_dim;
        float q0 = Q[q_base + d];
        float q1 = Q[q_base + d + half_dim];

        Q[q_base + d] = q0 * c - q1 * sn;
        Q[q_base + d + half_dim] = q0 * sn + q1 * c;
    }

    int total_k_pairs = batch_size * seq_len * num_kv_heads * half_dim;
    if (idx < total_k_pairs) {
        int d = idx % half_dim;
        int temp = idx / half_dim;
        int h = temp % num_kv_heads;
        temp = temp / num_kv_heads;
        int s = temp % seq_len;
        int b = temp / seq_len;

        float c = cos[s * half_dim + d];
        float sn = sin[s * half_dim + d];

        int k_base = ((b * seq_len + s) * num_kv_heads + h) * head_dim;
        float k0 = K[k_base + d];
        float k1 = K[k_base + d + half_dim];

        K[k_base + d] = k0 * c - k1 * sn;
        K[k_base + d + half_dim] = k0 * sn + k1 * c;
    }
}

} // namespace m2lrf
