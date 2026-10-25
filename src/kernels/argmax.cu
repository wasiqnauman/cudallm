#include <climits>

#include "common/cuda_check.h"
#include "kernels/kernels.h"

namespace cudallm {

namespace {

constexpr int kArgmaxThreads = 256;

// Larger value wins; on equal values the lower index wins (matches torch/numpy argmax).
__device__ __forceinline__ bool better(float v, int i, float bv, int bi) {
    return v > bv || (v == bv && i < bi);
}

// Shared-memory tree reduction of (value, index) pairs over a 256-thread block; the result
// lands in sv[0], si[0]. Deterministic: the pairing order is fixed.
__device__ __forceinline__ void block_argmax(float* sv, int* si) {
    for (int stride = kArgmaxThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const int o = threadIdx.x + stride;
            if (better(sv[o], si[o], sv[threadIdx.x], si[threadIdx.x])) {
                sv[threadIdx.x] = sv[o];
                si[threadIdx.x] = si[o];
            }
        }
        __syncthreads();
    }
}

// argmax_partial_kernel: grid kArgmaxBlocks (256), block 256, static smem 256 floats + 256 ints.
// Grid-stride over n; each thread keeps its best (value, lowest index); block reduction;
// thread 0 writes the block's partial result to part_v/part_i[blockIdx.x].
__global__ void argmax_partial_kernel(const float* __restrict__ logits, int n, float* __restrict__ part_v,
                                      int* __restrict__ part_i) {
    __shared__ float sv[kArgmaxThreads];
    __shared__ int si[kArgmaxThreads];
    float bv = -INFINITY;
    int bi = INT_MAX;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float v = logits[i];
        if (better(v, i, bv, bi)) {
            bv = v;
            bi = i;
        }
    }
    sv[threadIdx.x] = bv;
    si[threadIdx.x] = bi;
    __syncthreads();
    block_argmax(sv, si);
    if (threadIdx.x == 0) {
        part_v[blockIdx.x] = sv[0];
        part_i[blockIdx.x] = si[0];
    }
}

// argmax_final_kernel: grid 1, block 256, static smem 256 floats + 256 ints.
// Reduces the 256 partials with the same tie rule; thread 0 commits the token and advances
// the device position, so the next step needs no host involvement.
__global__ void argmax_final_kernel(const float* __restrict__ part_v, const int* __restrict__ part_i,
                                    int* __restrict__ d_token, int* __restrict__ d_pos) {
    __shared__ float sv[kArgmaxThreads];
    __shared__ int si[kArgmaxThreads];
    sv[threadIdx.x] = part_v[threadIdx.x];
    si[threadIdx.x] = part_i[threadIdx.x];
    __syncthreads();
    block_argmax(sv, si);
    if (threadIdx.x == 0) {
        *d_token = si[0];
        *d_pos += 1;
    }
}

}  // namespace

void launch_argmax(const float* logits, int n, float* part_v, int* part_i, int* d_token, int* d_pos, cudaStream_t stream) {
    CUDALLM_CHECK(logits && part_v && part_i && d_token && d_pos, "launch_argmax: null pointer");
    CUDALLM_CHECK(n > 0, "launch_argmax: n must be positive");
    static_assert(kArgmaxBlocks == kArgmaxThreads, "final reduction assumes one partial per thread");
    argmax_partial_kernel<<<kArgmaxBlocks, kArgmaxThreads, 0, stream>>>(logits, n, part_v, part_i);
    CUDALLM_CUDA_CHECK(cudaGetLastError());
    argmax_final_kernel<<<1, kArgmaxThreads, 0, stream>>>(part_v, part_i, d_token, d_pos);
    CUDALLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace cudallm
