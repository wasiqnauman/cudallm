#pragma once

#include <cstdint>

namespace cudallm {

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// Requires blockDim.x a multiple of 32 and <= 1024; smem32 holds 32 floats.
__device__ __forceinline__ float block_reduce_sum(float v, float* smem32) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int n_warps = blockDim.x / 32;
    v = warp_reduce_sum(v);
    if (lane == 0) smem32[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float w = lane < n_warps ? smem32[lane] : 0.f;
        w = warp_reduce_sum(w);
        if (lane == 0) smem32[0] = w;
    }
    __syncthreads();
    const float r = smem32[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }

__device__ __forceinline__ void unpack_bf16x2(uint32_t w, float& lo, float& hi) {
    lo = __uint_as_float(w << 16);
    hi = __uint_as_float(w & 0xffff0000u);
}

__device__ __forceinline__ float silu(float g) { return g / (1.f + expf(-g)); }

}  // namespace cudallm
