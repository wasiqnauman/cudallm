#include "common/cuda_check.h"
#include "kernels/kernels.h"

namespace cudallm {

void launch_matvec_naive(const MatvecArgs&, cudaStream_t) {
    throw std::runtime_error("not implemented: launch_matvec_naive");
}

}  // namespace cudallm
