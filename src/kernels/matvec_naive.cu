#include "common/cuda_check.h"
#include "kernels/common.cuh"
#include "kernels/kernels.h"

namespace cudallm {

namespace {

// matvec_naive_kernel: grid ceil(rows/256), block 256, no shared memory.
// One thread per output row, looping over all columns in order with an fp32 accumulator.
// Neighbouring threads read different rows, so loads are uncoalesced: this is the
// deliberate baseline (DESIGN.md §10.5) and must not be improved.
__global__ void matvec_naive_kernel(const uint16_t* __restrict__ w, const float* __restrict__ x,
                                    float* __restrict__ y, int rows, int cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const uint16_t* wr = w + static_cast<size_t>(row) * cols;
    float acc = 0.f;
    for (int j = 0; j < cols; ++j) acc += bf16_bits_to_float(wr[j]) * x[j];
    y[row] = acc;
}

}  // namespace

void launch_matvec_naive(const MatvecArgs& a, cudaStream_t stream) {
    CUDALLM_CHECK(a.w && a.x && a.y, "launch_matvec_naive: null pointer");
    CUDALLM_CHECK(a.rows > 0 && a.cols > 0, "launch_matvec_naive: bad shape");
    CUDALLM_CHECK(a.norm_w == nullptr, "launch_matvec_naive: fused norm not supported");
    CUDALLM_CHECK(static_cast<const void*>(a.x) != static_cast<const void*>(a.y), "launch_matvec_naive: x and y alias");
    matvec_naive_kernel<<<(a.rows + 255) / 256, 256, 0, stream>>>(static_cast<const uint16_t*>(a.w), a.x, a.y, a.rows, a.cols);
    CUDALLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace cudallm
