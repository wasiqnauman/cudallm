#include <cuda_fp16.h>

#include "common/cuda_check.h"
#include "kernels/kernels.h"

namespace cudallm {

namespace {

// rope_inplace_kernel: grid nh + nkv, block hd/2, no shared memory.
// Block b < nh rotates q head b, otherwise k head b-nh, in place in qkv. Thread i rotates
// the pair (i, i+hd/2) by angle pos*inv_freq[i] (HF rotate_half convention).
__global__ void rope_inplace_kernel(float* __restrict__ qkv, const float* __restrict__ inv_freq,
                                    const int* __restrict__ d_pos, int nh, int hd) {
    const int b = blockIdx.x;
    const int i = threadIdx.x;
    const int half = hd / 2;
    float* v = (b < nh) ? qkv + b * hd : qkv + nh * hd + (b - nh) * hd;
    const float ang = static_cast<float>(*d_pos) * inv_freq[i];
    float s, c;
    sincosf(ang, &s, &c);
    const float x0 = v[i];
    const float x1 = v[i + half];
    v[i] = x0 * c - x1 * s;
    v[i + half] = x1 * c + x0 * s;
}

// kv_write_kernel: grid 2*nkv, block hd, no shared memory.
// Block b < nkv copies k head b, otherwise v head b-nkv, from qkv into its cache at
// [(head*max_seq + pos)*hd + d] as fp16 (round to nearest). Thread d copies element d.
__global__ void kv_write_kernel(const float* __restrict__ qkv, __half* __restrict__ k_cache,
                                __half* __restrict__ v_cache, const int* __restrict__ d_pos, int nh, int nkv,
                                int hd, int max_seq) {
    const int b = blockIdx.x;
    const int d = threadIdx.x;
    const int pos = *d_pos;
    const bool is_k = b < nkv;
    const int head = is_k ? b : b - nkv;
    const float* src = qkv + (nh + (is_k ? 0 : nkv) + head) * hd;
    __half* cache = is_k ? k_cache : v_cache;
    cache[(static_cast<size_t>(head) * max_seq + pos) * hd + d] = __float2half_rn(src[d]);
}

}  // namespace

void launch_rope_inplace(float* qkv, const float* inv_freq, const int* d_pos, int nh, int nkv, int hd, cudaStream_t stream) {
    CUDALLM_CHECK(qkv && inv_freq && d_pos, "launch_rope_inplace: null pointer");
    CUDALLM_CHECK(nh > 0 && nkv > 0 && hd > 0 && hd % 2 == 0 && hd <= 2048, "launch_rope_inplace: bad dimensions");
    rope_inplace_kernel<<<nh + nkv, hd / 2, 0, stream>>>(qkv, inv_freq, d_pos, nh, hd);
    CUDALLM_CUDA_CHECK(cudaGetLastError());
}

void launch_kv_write(const float* qkv, __half* k_cache_l, __half* v_cache_l, const int* d_pos, int nh, int nkv, int hd,
                     int max_seq, cudaStream_t stream) {
    CUDALLM_CHECK(qkv && k_cache_l && v_cache_l && d_pos, "launch_kv_write: null pointer");
    CUDALLM_CHECK(nh > 0 && nkv > 0 && hd > 0 && hd <= 1024 && max_seq > 0, "launch_kv_write: bad dimensions");
    kv_write_kernel<<<2 * nkv, hd, 0, stream>>>(qkv, k_cache_l, v_cache_l, d_pos, nh, nkv, hd, max_seq);
    CUDALLM_CUDA_CHECK(cudaGetLastError());
}

void launch_rope_kv_fused(float*, const float*, __half*, __half*, const int*, int, int, int, int, cudaStream_t) {
    throw std::runtime_error("not implemented: launch_rope_kv_fused");
}

}  // namespace cudallm
