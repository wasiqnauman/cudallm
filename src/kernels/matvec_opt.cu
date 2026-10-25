#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <unordered_map>

#include "common/cuda_check.h"
#include "kernels/common.cuh"
#include "kernels/kernels.h"

namespace cudallm {

int default_rpb(WeightType w) {
    switch (w) {
        case WeightType::BF16: return kDefaultRpbBf16;
        case WeightType::INT8: return kDefaultRpbInt8;
        case WeightType::INT4: return kDefaultRpbInt4;
    }
    throw std::runtime_error("default_rpb: unknown weight type");
}

namespace {

// Dot product of weight row `r` with the staged input xs, for one warp. Each lane issues
// 128-bit loads; per iteration the warp covers 256 (bf16), 512 (int8) or 1024 (int4) columns.
// Quantized rows apply their scales here (int8: once after the loop; int4: per 128-col group).
template <WeightType WT>
__device__ __forceinline__ float row_dot(const MatvecArgs& a, int r, const float* __restrict__ xs, int lane) {
    float acc = 0.f;
    if constexpr (WT == WeightType::BF16) {
        const uint4* row = reinterpret_cast<const uint4*>(static_cast<const uint8_t*>(a.w) + static_cast<size_t>(r) * a.cols * 2);
        const int iters = a.cols / 256;
#pragma unroll 4
        for (int it = 0; it < iters; ++it) {
            const uint4 v = row[it * 32 + lane];
            const int base = it * 256 + lane * 8;
            const float4 x0 = *reinterpret_cast<const float4*>(xs + base);
            const float4 x1 = *reinterpret_cast<const float4*>(xs + base + 4);
            float lo, hi;
            unpack_bf16x2(v.x, lo, hi);
            acc += lo * x0.x + hi * x0.y;
            unpack_bf16x2(v.y, lo, hi);
            acc += lo * x0.z + hi * x0.w;
            unpack_bf16x2(v.z, lo, hi);
            acc += lo * x1.x + hi * x1.y;
            unpack_bf16x2(v.w, lo, hi);
            acc += lo * x1.z + hi * x1.w;
        }
    } else if constexpr (WT == WeightType::INT8) {
        const uint4* row = reinterpret_cast<const uint4*>(static_cast<const uint8_t*>(a.w) + static_cast<size_t>(r) * a.cols);
        const int iters = a.cols / 512;
#pragma unroll 4
        for (int it = 0; it < iters; ++it) {
            const uint4 v = row[it * 32 + lane];
            const int base = it * 512 + lane * 16;
            const uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float4 x = *reinterpret_cast<const float4*>(xs + base + 4 * k);
                const uint32_t w = words[k];
                acc += static_cast<float>(static_cast<int8_t>(w & 0xff)) * x.x;
                acc += static_cast<float>(static_cast<int8_t>((w >> 8) & 0xff)) * x.y;
                acc += static_cast<float>(static_cast<int8_t>((w >> 16) & 0xff)) * x.z;
                acc += static_cast<float>(static_cast<int8_t>(w >> 24)) * x.w;
            }
        }
        acc *= static_cast<const float*>(a.scales)[r];
    } else {
        const uint4* row = reinterpret_cast<const uint4*>(static_cast<const uint8_t*>(a.w) + static_cast<size_t>(r) * a.cols / 2);
        const __half* scales = static_cast<const __half*>(a.scales) + static_cast<size_t>(r) * (a.cols / 128);
        const int iters = a.cols / 1024;
#pragma unroll 4
        for (int it = 0; it < iters; ++it) {
            const uint4 v = row[it * 32 + lane];
            const int base = it * 1024 + lane * 32;
            const uint32_t words[4] = {v.x, v.y, v.z, v.w};
            float part = 0.f;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t w = words[k];
                const float4 x0 = *reinterpret_cast<const float4*>(xs + base + 8 * k);
                const float4 x1 = *reinterpret_cast<const float4*>(xs + base + 8 * k + 4);
                // Byte b holds element 2b in its low nibble and 2b+1 in its high nibble, stored as q+8.
                part += static_cast<float>(static_cast<int>(w & 0xf) - 8) * x0.x;
                part += static_cast<float>(static_cast<int>((w >> 4) & 0xf) - 8) * x0.y;
                part += static_cast<float>(static_cast<int>((w >> 8) & 0xf) - 8) * x0.z;
                part += static_cast<float>(static_cast<int>((w >> 12) & 0xf) - 8) * x0.w;
                part += static_cast<float>(static_cast<int>((w >> 16) & 0xf) - 8) * x1.x;
                part += static_cast<float>(static_cast<int>((w >> 20) & 0xf) - 8) * x1.y;
                part += static_cast<float>(static_cast<int>((w >> 24) & 0xf) - 8) * x1.z;
                part += static_cast<float>(static_cast<int>(w >> 28) - 8) * x1.w;
            }
            acc += part * __half2float(scales[base / 128]);
        }
    }
    return acc;
}

// matvec_opt_kernel<WT, EP, FUSE_NORM, RPB>
// Grid: persistent, min(ceil(out_rows/RPB), num_SMs * max_active_blocks_per_SM), grid-stride over rows.
// Block: RPB warps (RPB*32 threads). Dynamic smem: cols floats (staged input xs) + static 32 floats.
// Prologue: the block stages x (RMS-normalized and scaled by norm_w when FUSE_NORM) into xs.
// Main loop: warp w handles rows blockIdx.x*RPB + w, + gridDim.x*RPB, ...; each lane does 128-bit
// loads of the row, the warp reduces, lane 0 applies the epilogue (STORE, ADD, or SWIGLU which
// pairs gate row r with up row r + rows/2).
template <WeightType WT, Epilogue EP, bool FUSE_NORM, int RPB>
__global__ void __launch_bounds__(RPB * 32) matvec_opt_kernel(MatvecArgs a) {
    extern __shared__ float xs[];
    __shared__ float red[32];
    const int tid = threadIdx.x;
    if constexpr (FUSE_NORM) {
        float ss = 0.f;
        for (int j = tid; j < a.cols; j += blockDim.x) ss += a.x[j] * a.x[j];
        ss = block_reduce_sum(ss, red);
        const float inv = rsqrtf(ss / static_cast<float>(a.cols) + a.eps);
        for (int j = tid; j < a.cols; j += blockDim.x) xs[j] = a.x[j] * inv * a.norm_w[j];
    } else {
        for (int j = tid; j < a.cols; j += blockDim.x) xs[j] = a.x[j];
    }
    __syncthreads();

    const int warp = tid / 32;
    const int lane = tid % 32;
    const int out_rows = (EP == Epilogue::SWIGLU) ? a.rows / 2 : a.rows;
    for (int r = blockIdx.x * RPB + warp; r < out_rows; r += gridDim.x * RPB) {
        float d = warp_reduce_sum(row_dot<WT>(a, r, xs, lane));
        if constexpr (EP == Epilogue::SWIGLU) {
            const float up = warp_reduce_sum(row_dot<WT>(a, r + out_rows, xs, lane));
            if (lane == 0) a.y[r] = silu(d) * up;
        } else if constexpr (EP == Epilogue::ADD) {
            if (lane == 0) a.y[r] += d;
        } else {
            if (lane == 0) a.y[r] = d;
        }
    }
}

int num_sms() {
    static int n = [] {
        int dev = 0, v = 0;
        CUDALLM_CUDA_CHECK(cudaGetDevice(&dev));
        CUDALLM_CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev));
        return v;
    }();
    return n;
}

template <WeightType WT, Epilogue EP, bool FUSE_NORM, int RPB>
void launch_instance(const MatvecArgs& a, cudaStream_t stream) {
    auto kernel = matvec_opt_kernel<WT, EP, FUSE_NORM, RPB>;
    const size_t smem = static_cast<size_t>(a.cols) * sizeof(float);
    // Occupancy depends only on the instantiation and smem size (i.e. cols); cache it.
    static std::unordered_map<int, int> max_active_by_cols;
    auto it = max_active_by_cols.find(a.cols);
    if (it == max_active_by_cols.end()) {
        int blocks = 0;
        CUDALLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, RPB * 32, smem));
        CUDALLM_CHECK(blocks > 0, "launch_matvec_opt: kernel cannot be resident (occupancy 0)");
        it = max_active_by_cols.emplace(a.cols, blocks).first;
    }
    const int out_rows = (EP == Epilogue::SWIGLU) ? a.rows / 2 : a.rows;
    const int needed = (out_rows + RPB - 1) / RPB;
    const int grid = std::min(needed, num_sms() * it->second);
    kernel<<<grid, RPB * 32, smem, stream>>>(a);
    CUDALLM_CUDA_CHECK(cudaGetLastError());
}

template <WeightType WT, Epilogue EP, bool FUSE_NORM>
void dispatch_rpb(int rpb, const MatvecArgs& a, cudaStream_t s) {
    switch (rpb) {
        case 1: launch_instance<WT, EP, FUSE_NORM, 1>(a, s); return;
        case 2: launch_instance<WT, EP, FUSE_NORM, 2>(a, s); return;
        case 4: launch_instance<WT, EP, FUSE_NORM, 4>(a, s); return;
        case 8: launch_instance<WT, EP, FUSE_NORM, 8>(a, s); return;
    }
    throw std::runtime_error("launch_matvec_opt: rpb must be 1, 2, 4 or 8");
}

template <WeightType WT, Epilogue EP>
void dispatch_norm(bool fuse_norm, int rpb, const MatvecArgs& a, cudaStream_t s) {
    if (fuse_norm) {
        dispatch_rpb<WT, EP, true>(rpb, a, s);
    } else {
        dispatch_rpb<WT, EP, false>(rpb, a, s);
    }
}

template <WeightType WT>
void dispatch_ep(Epilogue ep, bool fuse_norm, int rpb, const MatvecArgs& a, cudaStream_t s) {
    switch (ep) {
        case Epilogue::STORE: dispatch_norm<WT, Epilogue::STORE>(fuse_norm, rpb, a, s); return;
        case Epilogue::ADD: dispatch_norm<WT, Epilogue::ADD>(fuse_norm, rpb, a, s); return;
        case Epilogue::SWIGLU: dispatch_norm<WT, Epilogue::SWIGLU>(fuse_norm, rpb, a, s); return;
    }
    throw std::runtime_error("launch_matvec_opt: unknown epilogue");
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

}  // namespace

void launch_matvec_opt(WeightType wt, Epilogue ep, bool fuse_norm, const MatvecArgs& a, int rpb, cudaStream_t stream) {
    CUDALLM_CHECK(a.w && a.x && a.y, "launch_matvec_opt: null pointer");
    CUDALLM_CHECK(a.rows > 0 && a.cols > 0, "launch_matvec_opt: bad shape");
    CUDALLM_CHECK(a.cols % 1024 == 0, "launch_matvec_opt: cols must be a multiple of 1024");
    CUDALLM_CHECK(static_cast<size_t>(a.cols) * 4 <= 48 * 1024, "launch_matvec_opt: cols*4 must fit in 48 KiB of shared memory");
    CUDALLM_CHECK(ep != Epilogue::SWIGLU || a.rows % 2 == 0, "launch_matvec_opt: SWIGLU needs an even row count");
    CUDALLM_CHECK(rpb == 1 || rpb == 2 || rpb == 4 || rpb == 8, "launch_matvec_opt: rpb must be 1, 2, 4 or 8");
    CUDALLM_CHECK(aligned16(a.w) && aligned16(a.x) && aligned16(a.y), "launch_matvec_opt: pointers must be 16-byte aligned");
    CUDALLM_CHECK(!fuse_norm || a.norm_w, "launch_matvec_opt: fuse_norm needs norm_w");
    CUDALLM_CHECK(wt == WeightType::BF16 || a.scales, "launch_matvec_opt: quantized weights need scales");
    CUDALLM_CHECK(static_cast<const void*>(a.x) != static_cast<const void*>(a.y), "launch_matvec_opt: x and y alias");
    switch (wt) {
        case WeightType::BF16: dispatch_ep<WeightType::BF16>(ep, fuse_norm, rpb, a, stream); return;
        case WeightType::INT8: dispatch_ep<WeightType::INT8>(ep, fuse_norm, rpb, a, stream); return;
        case WeightType::INT4: dispatch_ep<WeightType::INT4>(ep, fuse_norm, rpb, a, stream); return;
    }
    throw std::runtime_error("launch_matvec_opt: unknown weight type");
}

}  // namespace cudallm
