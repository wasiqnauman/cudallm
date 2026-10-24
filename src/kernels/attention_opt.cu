#include "common/cuda_check.h"
#include "kernels/kernels.h"

namespace cudallm {

void launch_attention_opt(const float*, const __half*, const __half*, const int*, float*, float*, float*, int, int, int, int, cudaStream_t) {
    throw std::runtime_error("not implemented: launch_attention_opt");
}

}  // namespace cudallm
