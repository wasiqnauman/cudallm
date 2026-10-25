#include "common/cuda_check.h"
#include "kernels/common.cuh"
#include "kernels/kernels.h"

namespace cudallm {

namespace {

// rmsnorm_kernel: grid 1, block 256, static smem 32 floats (reduction scratch).
// Each thread sums squares of x[t], x[t+256], ...; block reduction; then each thread
// writes y[j] = x[j] * rsqrt(mean(x^2) + eps) * w[j] for its strided j.
__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ w, float eps,
                               float* __restrict__ y, int n) {
    __shared__ float smem32[32];
    float ss = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) ss += x[j] * x[j];
    ss = block_reduce_sum(ss, smem32);
    const float inv = rsqrtf(ss / static_cast<float>(n) + eps);
    for (int j = threadIdx.x; j < n; j += blockDim.x) y[j] = x[j] * inv * w[j];
}

}  // namespace

void launch_rmsnorm(const float* x, const float* w, float eps, float* y, int n, cudaStream_t stream) {
    CUDALLM_CHECK(x && w && y, "launch_rmsnorm: null pointer");
    CUDALLM_CHECK(n > 0 && n % 256 == 0, "launch_rmsnorm: n must be a positive multiple of 256");
    rmsnorm_kernel<<<1, 256, 0, stream>>>(x, w, eps, y, n);
    CUDALLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace cudallm
