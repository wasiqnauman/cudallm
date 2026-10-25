#include "common/cuda_check.h"
#include "kernels/common.cuh"
#include "kernels/kernels.h"

namespace cudallm {

namespace {

// add_inplace_kernel: grid ceil(n/256), block 256. Thread i: x[i] += y[i].
__global__ void add_inplace_kernel(float* __restrict__ x, const float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// silu_mul_kernel: grid ceil(inter/256), block 256. Thread i: act[i] = silu(gate[i]) * up[i],
// where gu holds gate in [0, inter) and up in [inter, 2*inter).
__global__ void silu_mul_kernel(const float* __restrict__ gu, float* __restrict__ act, int inter) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inter) act[i] = silu(gu[i]) * gu[i + inter];
}

// set_token_kernel: grid 1, block 1. Writes the token from a kernel argument so the host
// never shares a buffer with an in-flight copy.
__global__ void set_token_kernel(int* d_token, int token) { *d_token = token; }

}  // namespace

void launch_add_inplace(float* x, const float* y, int n, cudaStream_t stream) {
    CUDALLM_CHECK(x && y && n > 0, "launch_add_inplace: bad arguments");
    add_inplace_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, y, n);
    CUDALLM_CUDA_CHECK(cudaGetLastError());
}

void launch_silu_mul(const float* gu, float* act, int inter, cudaStream_t stream) {
    CUDALLM_CHECK(gu && act && inter > 0, "launch_silu_mul: bad arguments");
    silu_mul_kernel<<<(inter + 255) / 256, 256, 0, stream>>>(gu, act, inter);
    CUDALLM_CUDA_CHECK(cudaGetLastError());
}

void launch_set_token(int* d_token, int token, cudaStream_t stream) {
    CUDALLM_CHECK(d_token && token >= 0, "launch_set_token: bad arguments");
    set_token_kernel<<<1, 1, 0, stream>>>(d_token, token);
    CUDALLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace cudallm
