#include "common/cuda_check.h"
#include "kernels/kernels.h"

namespace cudallm {

int default_rpb(WeightType w) {
    switch (w) {
        case WeightType::BF16: return kDefaultRpbBf16;
        case WeightType::INT8: return kDefaultRpbInt8;
        case WeightType::INT4: return kDefaultRpbInt4;
    }
    throw std::runtime_error("default_rpb: unknown weight type");
}

void launch_matvec_opt(WeightType, Epilogue, bool, const MatvecArgs&, int, cudaStream_t) {
    throw std::runtime_error("not implemented: launch_matvec_opt");
}

}  // namespace cudallm
