#include "common/cuda_check.h"
#include "kernels/kernels.h"

namespace cudallm {

void launch_rmsnorm(const float*, const float*, float, float*, int, cudaStream_t) {
    throw std::runtime_error("not implemented: launch_rmsnorm");
}

}  // namespace cudallm
