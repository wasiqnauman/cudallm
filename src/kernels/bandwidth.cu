#include "common/cuda_check.h"
#include "kernels/kernels.h"

namespace cudallm {

void launch_bw_read(const uint4*, size_t, unsigned*, int, cudaStream_t) {
    throw std::runtime_error("not implemented: launch_bw_read");
}

void launch_bw_write(uint4*, size_t, int, cudaStream_t) {
    throw std::runtime_error("not implemented: launch_bw_write");
}

void launch_bw_copy(const uint4*, uint4*, size_t, int, cudaStream_t) {
    throw std::runtime_error("not implemented: launch_bw_copy");
}

}  // namespace cudallm
