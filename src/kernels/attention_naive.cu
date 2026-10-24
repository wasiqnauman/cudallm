#include "common/cuda_check.h"
#include "kernels/kernels.h"

namespace cudallm {

void launch_attention_naive(const float*, const __half*, const __half*, const int*, float*, int, int, int, int, cudaStream_t) {
    throw std::runtime_error("not implemented: launch_attention_naive");
}

}  // namespace cudallm
