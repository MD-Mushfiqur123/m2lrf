#include "../include/m2lrf_common.cuh"

namespace m2lrf {

// Fused SwiGLU Activation Forward: Y = (Gate * sigmoid(Gate)) * Up
__global__ void fast_swiglu_forward_kernel(
    const float *__restrict__ Gate,
    const float *__restrict__ Up,
    float *__restrict__ Y,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float g = Gate[idx];
        float u = Up[idx];
        float silu_g = g / (1.0f + expf(-g));
        Y[idx] = silu_g * u;
    }
}

// Fused SwiGLU Backward: dGate, dUp from dY
__global__ void fast_swiglu_backward_kernel(
    const float *__restrict__ dY,
    const float *__restrict__ Gate,
    const float *__restrict__ Up,
    float *__restrict__ dGate,
    float *__restrict__ dUp,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float dy = dY[idx];
        float g = Gate[idx];
        float u = Up[idx];

        float sig = 1.0f / (1.0f + expf(-g));
        float silu_g = g * sig;
        
        // dUp = dY * silu(g)
        dUp[idx] = dy * silu_g;
        
        // dGate = dY * u * sig * (1 + g * (1 - sig))
        dGate[idx] = dy * u * sig * (1.0f + g * (1.0f - sig));
    }
}

} // namespace m2lrf
