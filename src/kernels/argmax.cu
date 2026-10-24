#include "common/cuda_check.h"
#include "kernels/kernels.h"

namespace cudallm {

void launch_argmax(const float*, int, float*, int*, int*, int*, cudaStream_t) {
    throw std::runtime_error("not implemented: launch_argmax");
}

}  // namespace cudallm
