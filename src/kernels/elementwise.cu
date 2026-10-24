#include "common/cuda_check.h"
#include "kernels/kernels.h"

namespace cudallm {

void launch_add_inplace(float*, const float*, int, cudaStream_t) {
    throw std::runtime_error("not implemented: launch_add_inplace");
}

void launch_silu_mul(const float*, float*, int, cudaStream_t) {
    throw std::runtime_error("not implemented: launch_silu_mul");
}

void launch_set_token(int*, int, cudaStream_t) {
    throw std::runtime_error("not implemented: launch_set_token");
}

}  // namespace cudallm
