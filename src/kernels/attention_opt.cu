#include <cuda_fp16.h>

#include "common/cuda_check.h"
#include "kernels/common.cuh"
#include "kernels/kernels.h"

namespace cudallm {

namespace {

constexpr int kGroup = 4;     // q heads per kv head
constexpr int kHeadDim = 64;
constexpr int kChunk = kAttnChunk;  // tokens per split
constexpr int kPartialThreads = kGroup * 32;
constexpr int kKsStride = kHeadDim + 2;  // padded row (halves) so lane-per-row reads hit distinct banks

// attention_partial_kernel: grid (nkv, max_seq/128), block 128 (4 warps), static smem
// Ks[128][66] + Vs[128][64] halves, qs[4][64] + ps[4][128] floats (35.5 KiB).
// Block (kvh, split) loads the K/V tile for tokens [split*128, split*128+n) of kv head kvh once and
// shares it across the 4 q heads of the GQA group (warp w = q head kvh*4+w). Each lane scores
// tokens lane, lane+32, lane+64, lane+96; the warp forms the split-local max m and sum l; then lane
// owns output dims 2*lane, 2*lane+1 and accumulates the unnormalized p·V. Splits past the context
// exit immediately, so the grid is fixed (graph-capturable) while the work scales with ctx.
__global__ void __launch_bounds__(kPartialThreads)
attention_partial_kernel(const float* __restrict__ q, const __half* __restrict__ k_cache, const __half* __restrict__ v_cache,
                         const int* __restrict__ d_pos, float* __restrict__ part_o, float* __restrict__ part_ml, int max_seq) {
    __shared__ __half Ks[kChunk][kKsStride];
    __shared__ __align__(16) __half Vs[kChunk][kHeadDim];
    __shared__ float qs[kGroup][kHeadDim];
    __shared__ float ps[kGroup][kChunk];

    const int ctx = *d_pos + 1;
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int t0 = split * kChunk;
    if (t0 >= ctx) return;
    const int n = min(kChunk, ctx - t0);
    const int tid = threadIdx.x;

    // Each token row is 64 halves = 8 uint4; the tile is loaded as n*8 128-bit chunks.
    const __half* Kg = k_cache + (static_cast<size_t>(kvh) * max_seq + t0) * kHeadDim;
    const __half* Vg = v_cache + (static_cast<size_t>(kvh) * max_seq + t0) * kHeadDim;
    for (int idx = tid; idx < n * 8; idx += kPartialThreads) {
        const int t = idx / 8;
        const int c = idx % 8;
        const uint4 kv = reinterpret_cast<const uint4*>(Kg + t * kHeadDim)[c];
        const uint4 vv = reinterpret_cast<const uint4*>(Vg + t * kHeadDim)[c];
        __half2* kdst = reinterpret_cast<__half2*>(&Ks[t][c * 8]);
        kdst[0] = *reinterpret_cast<const __half2*>(&kv.x);
        kdst[1] = *reinterpret_cast<const __half2*>(&kv.y);
        kdst[2] = *reinterpret_cast<const __half2*>(&kv.z);
        kdst[3] = *reinterpret_cast<const __half2*>(&kv.w);
        *reinterpret_cast<uint4*>(&Vs[t][c * 8]) = vv;
    }
    for (int idx = tid; idx < kGroup * kHeadDim; idx += kPartialThreads) {
        qs[idx / kHeadDim][idx % kHeadDim] = q[(kvh * kGroup) * kHeadDim + idx];
    }
    __syncthreads();

    const int w = tid / 32;
    const int lane = tid % 32;
    const int h = kvh * kGroup + w;
    float s[4];
    float local_max = -INFINITY;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int j = lane + 32 * k;
        if (j < n) {
            float dot = 0.f;
            const __half2* krow = reinterpret_cast<const __half2*>(&Ks[j][0]);
#pragma unroll 8
            for (int d2 = 0; d2 < kHeadDim / 2; ++d2) {
                const float2 kf = __half22float2(krow[d2]);
                dot += kf.x * qs[w][2 * d2] + kf.y * qs[w][2 * d2 + 1];
            }
            s[k] = 0.125f * dot;
        } else {
            s[k] = -INFINITY;
        }
        local_max = fmaxf(local_max, s[k]);
    }
    const float m = warp_reduce_max(local_max);
    float local_sum = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int j = lane + 32 * k;
        const float p = (j < n) ? expf(s[k] - m) : 0.f;
        ps[w][j] = p;
        local_sum += p;
    }
    const float l = warp_reduce_sum(local_sum);
    __syncwarp();

    float o0 = 0.f, o1 = 0.f;
    for (int j = 0; j < n; ++j) {
        const float p = ps[w][j];
        const float2 vf = __half22float2(*reinterpret_cast<const __half2*>(&Vs[j][2 * lane]));
        o0 += p * vf.x;
        o1 += p * vf.y;
    }
    const int s_split = max_seq / kChunk;
    float* po = part_o + (static_cast<size_t>(h) * s_split + split) * kHeadDim;
    po[2 * lane] = o0;
    po[2 * lane + 1] = o1;
    if (lane == 0) {
        float* pml = part_ml + (static_cast<size_t>(h) * s_split + split) * 2;
        pml[0] = m;
        pml[1] = l;
    }
}

// attention_combine_kernel: grid nh, block 64, no smem. Thread d of block h merges the active
// splits in order (deterministic): M = max m_s, L = sum l_s*exp(m_s-M),
// out[h*64+d] = sum part_o[s][d]*exp(m_s-M) / L.
__global__ void __launch_bounds__(kHeadDim)
attention_combine_kernel(const float* __restrict__ part_o, const float* __restrict__ part_ml, const int* __restrict__ d_pos,
                         float* __restrict__ out, int max_seq) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    const int ctx = *d_pos + 1;
    const int n_split = (ctx + kChunk - 1) / kChunk;
    const int s_split = max_seq / kChunk;
    const float* ml = part_ml + static_cast<size_t>(h) * s_split * 2;
    const float* po = part_o + static_cast<size_t>(h) * s_split * kHeadDim;
    float M = -INFINITY;
    for (int s = 0; s < n_split; ++s) M = fmaxf(M, ml[2 * s]);
    float L = 0.f, acc = 0.f;
    for (int s = 0; s < n_split; ++s) {
        const float e = expf(ml[2 * s] - M);
        L += ml[2 * s + 1] * e;
        acc += po[static_cast<size_t>(s) * kHeadDim + d] * e;
    }
    out[h * kHeadDim + d] = acc / L;
}

}  // namespace

void launch_attention_opt(const float* q, const __half* k_l, const __half* v_l, const int* d_pos, float* part_o,
                          float* part_ml, float* out, int nh, int nkv, int hd, int max_seq, cudaStream_t stream) {
    CUDALLM_CHECK(q && k_l && v_l && d_pos && part_o && part_ml && out, "launch_attention_opt: null pointer");
    CUDALLM_CHECK(nkv > 0 && nh == nkv * kGroup, "launch_attention_opt: requires nh/nkv == 4");
    CUDALLM_CHECK(hd == kHeadDim, "launch_attention_opt: requires hd == 64");
    CUDALLM_CHECK(max_seq > 0 && max_seq % kChunk == 0, "launch_attention_opt: max_seq must be a multiple of 128");
    CUDALLM_CHECK((reinterpret_cast<uintptr_t>(k_l) & 15u) == 0 && (reinterpret_cast<uintptr_t>(v_l) & 15u) == 0,
                  "launch_attention_opt: caches must be 16-byte aligned");
    attention_partial_kernel<<<dim3(nkv, max_seq / kChunk), kPartialThreads, 0, stream>>>(q, k_l, v_l, d_pos, part_o, part_ml,
                                                                                        max_seq);
    CUDALLM_CUDA_CHECK(cudaGetLastError());
    attention_combine_kernel<<<nh, kHeadDim, 0, stream>>>(part_o, part_ml, d_pos, out, max_seq);
    CUDALLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace cudallm
