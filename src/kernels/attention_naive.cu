#include <cuda_fp16.h>

#include "common/cuda_check.h"
#include "kernels/common.cuh"
#include "kernels/kernels.h"

namespace cudallm {

namespace {

constexpr int kAttnNaiveThreads = 256;

// attention_naive_kernel: grid nh, block 256, dynamic smem max_seq floats (scores) + 32 floats
// (reduction scratch). Block h handles q head h against kv head h/(nh/nkv), ctx = pos+1:
// (1) threads stride over tokens computing scaled scores, (2) block max, (3) exp and block sum,
// (4) threads d < hd each accumulate one output dim over all tokens. Every q head re-reads
// its K/V independently: the deliberate GQA inefficiency of the baseline.
__global__ void attention_naive_kernel(const float* __restrict__ q, const __half* __restrict__ k_cache,
                                       const __half* __restrict__ v_cache, const int* __restrict__ d_pos,
                                       float* __restrict__ out, int nh, int nkv, int hd, int max_seq) {
    extern __shared__ float smem[];
    float* s = smem;
    float* red = smem + max_seq;
    const int h = blockIdx.x;
    const int kvh = h / (nh / nkv);
    const int ctx = *d_pos + 1;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int n_warps = blockDim.x / 32;
    const float* qh = q + h * hd;
    const __half* K = k_cache + static_cast<size_t>(kvh) * max_seq * hd;
    const __half* V = v_cache + static_cast<size_t>(kvh) * max_seq * hd;

    float local_max = -INFINITY;
    for (int t = threadIdx.x; t < ctx; t += blockDim.x) {
        float dot = 0.f;
        for (int d = 0; d < hd; ++d) dot += qh[d] * __half2float(K[static_cast<size_t>(t) * hd + d]);
        s[t] = 0.125f * dot;
        local_max = fmaxf(local_max, s[t]);
    }

    // Block max: warp reduce, then warp 0 reduces the per-warp maxima.
    local_max = warp_reduce_max(local_max);
    if (lane == 0) red[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float m = lane < n_warps ? red[lane] : -INFINITY;
        m = warp_reduce_max(m);
        if (lane == 0) red[0] = m;
    }
    __syncthreads();
    const float m = red[0];
    __syncthreads();

    float local_sum = 0.f;
    for (int t = threadIdx.x; t < ctx; t += blockDim.x) {
        s[t] = expf(s[t] - m);
        local_sum += s[t];
    }
    const float sum = block_reduce_sum(local_sum, red);

    if (threadIdx.x < hd) {
        const int d = threadIdx.x;
        float acc = 0.f;
        for (int t = 0; t < ctx; ++t) acc += s[t] * __half2float(V[static_cast<size_t>(t) * hd + d]);
        out[h * hd + d] = acc / sum;
    }
}

}  // namespace

void launch_attention_naive(const float* q, const __half* k_l, const __half* v_l, const int* d_pos, float* out, int nh,
                            int nkv, int hd, int max_seq, cudaStream_t stream) {
    CUDALLM_CHECK(q && k_l && v_l && d_pos && out, "launch_attention_naive: null pointer");
    CUDALLM_CHECK(nh > 0 && nkv > 0 && nh % nkv == 0, "launch_attention_naive: nh must be a multiple of nkv");
    CUDALLM_CHECK(hd > 0 && hd <= kAttnNaiveThreads && max_seq > 0, "launch_attention_naive: bad dimensions");
    const size_t smem = (static_cast<size_t>(max_seq) + 32) * sizeof(float);
    CUDALLM_CHECK(smem <= 48 * 1024, "launch_attention_naive: max_seq too large for shared-memory scores");
    attention_naive_kernel<<<nh, kAttnNaiveThreads, smem, stream>>>(q, k_l, v_l, d_pos, out, nh, nkv, hd, max_seq);
    CUDALLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace cudallm
