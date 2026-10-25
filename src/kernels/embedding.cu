#include <cuda_fp16.h>

#include "common/cuda_check.h"
#include "kernels/common.cuh"
#include "kernels/kernels.h"

namespace cudallm {

namespace {

// embedding_kernel<WT>: grid hidden/256, block 256, no shared memory.
// Reads tok = *d_token on device (so the step can be graph-captured); thread j writes
// x[j] = dequantized emb[tok][j].
template <WeightType WT>
__global__ void embedding_kernel(const void* __restrict__ w, const void* __restrict__ scales,
                                 const int* __restrict__ d_token, float* __restrict__ x, int cols, int group_size) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t tok = static_cast<size_t>(*d_token);
    if constexpr (WT == WeightType::BF16) {
        x[j] = bf16_bits_to_float(static_cast<const uint16_t*>(w)[tok * cols + j]);
    } else if constexpr (WT == WeightType::INT8) {
        x[j] = static_cast<float>(static_cast<const int8_t*>(w)[tok * cols + j]) * static_cast<const float*>(scales)[tok];
    } else {
        const uint8_t b = static_cast<const uint8_t*>(w)[tok * (cols / 2) + j / 2];
        const int q = ((j & 1) ? (b >> 4) : (b & 0x0F)) - 8;
        const float s = __half2float(static_cast<const __half*>(scales)[tok * (cols / group_size) + j / group_size]);
        x[j] = static_cast<float>(q) * s;
    }
}

}  // namespace

void launch_embedding(const Linear& emb, const int* d_token, float* x, int hidden, int group_size, cudaStream_t stream) {
    CUDALLM_CHECK(emb.w && d_token && x, "launch_embedding: null pointer");
    CUDALLM_CHECK(emb.cols == hidden && hidden % 256 == 0, "launch_embedding: hidden must equal emb.cols and be a multiple of 256");
    const dim3 grid(hidden / 256), block(256);
    switch (emb.wtype) {
        case WeightType::BF16:
            embedding_kernel<WeightType::BF16><<<grid, block, 0, stream>>>(emb.w, emb.scales, d_token, x, emb.cols, group_size);
            break;
        case WeightType::INT8:
            CUDALLM_CHECK(emb.scales, "launch_embedding: int8 needs scales");
            embedding_kernel<WeightType::INT8><<<grid, block, 0, stream>>>(emb.w, emb.scales, d_token, x, emb.cols, group_size);
            break;
        case WeightType::INT4:
            CUDALLM_CHECK(emb.scales && group_size > 0 && hidden % group_size == 0, "launch_embedding: int4 needs scales and a valid group size");
            embedding_kernel<WeightType::INT4><<<grid, block, 0, stream>>>(emb.w, emb.scales, d_token, x, emb.cols, group_size);
            break;
    }
    CUDALLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace cudallm
