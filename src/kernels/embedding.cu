#include "common/cuda_check.h"
#include "kernels/kernels.h"

namespace cudallm {

void launch_embedding(const Linear&, const int*, float*, int, int, cudaStream_t) {
    throw std::runtime_error("not implemented: launch_embedding");
}

}  // namespace cudallm
