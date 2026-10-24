#include "common/cuda_check.h"
#include "kernels/kernels.h"

namespace cudallm {

void launch_rope_inplace(float*, const float*, const int*, int, int, int, cudaStream_t) {
    throw std::runtime_error("not implemented: launch_rope_inplace");
}

void launch_kv_write(const float*, __half*, __half*, const int*, int, int, int, int, cudaStream_t) {
    throw std::runtime_error("not implemented: launch_kv_write");
}

void launch_rope_kv_fused(float*, const float*, __half*, __half*, const int*, int, int, int, int, cudaStream_t) {
    throw std::runtime_error("not implemented: launch_rope_kv_fused");
}

}  // namespace cudallm
