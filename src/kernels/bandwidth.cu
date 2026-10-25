#include "common/cuda_check.h"
#include "kernels/kernels.h"

namespace cudallm {

namespace {

constexpr int kBwThreads = 256;

// bw_read_kernel: grid `grid` (num_SMs*32), block 256, static smem 8 words.
// Grid-stride 128-bit loads; each thread XORs every word it loads, the block XOR-reduces
// (warp shuffles, then thread 0 over the 8 warps) and writes one word to sink[blockIdx.x]
// so the loads cannot be eliminated.
__global__ void bw_read_kernel(const uint4* __restrict__ src, size_t n, unsigned* __restrict__ sink) {
    __shared__ unsigned warp_acc[kBwThreads / 32];
    unsigned acc = 0;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    for (int o = 16; o > 0; o >>= 1) acc ^= __shfl_xor_sync(0xffffffff, acc, o);
    if (threadIdx.x % 32 == 0) warp_acc[threadIdx.x / 32] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned r = 0;
        for (int w = 0; w < kBwThreads / 32; ++w) r ^= warp_acc[w];
        sink[blockIdx.x] = r;
    }
}

// bw_write_kernel: grid `grid`, block 256, no smem. Grid-stride 128-bit stores of (i,i,i,i).
__global__ void bw_write_kernel(uint4* __restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const unsigned u = static_cast<unsigned>(i);
        dst[i] = make_uint4(u, u, u, u);
    }
}

// bw_copy_kernel: grid `grid`, block 256, no smem. Grid-stride 128-bit dst[i] = src[i].
__global__ void bw_copy_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        dst[i] = src[i];
    }
}

}  // namespace

void launch_bw_read(const uint4* src, size_t n_uint4, unsigned* sink, int grid, cudaStream_t stream) {
    CUDALLM_CHECK(src && sink && n_uint4 > 0 && grid > 0, "launch_bw_read: bad arguments");
    bw_read_kernel<<<grid, kBwThreads, 0, stream>>>(src, n_uint4, sink);
    CUDALLM_CUDA_CHECK(cudaGetLastError());
}

void launch_bw_write(uint4* dst, size_t n_uint4, int grid, cudaStream_t stream) {
    CUDALLM_CHECK(dst && n_uint4 > 0 && grid > 0, "launch_bw_write: bad arguments");
    bw_write_kernel<<<grid, kBwThreads, 0, stream>>>(dst, n_uint4);
    CUDALLM_CUDA_CHECK(cudaGetLastError());
}

void launch_bw_copy(const uint4* src, uint4* dst, size_t n_uint4, int grid, cudaStream_t stream) {
    CUDALLM_CHECK(src && dst && n_uint4 > 0 && grid > 0, "launch_bw_copy: bad arguments");
    bw_copy_kernel<<<grid, kBwThreads, 0, stream>>>(src, dst, n_uint4);
    CUDALLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace cudallm
