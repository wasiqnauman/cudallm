#pragma once

#include "common/dtype.h"

namespace cudallm {

enum class MatvecImpl : int { NAIVE = 0, OPT = 1 };
enum class AttnImpl   : int { NAIVE = 0, OPT = 1 };

struct EngineOptions {
    MatvecImpl matvec = MatvecImpl::OPT;
    AttnImpl   attn   = AttnImpl::OPT;
    bool fuse = true;
    bool cuda_graph = true;
    int  max_seq_len = 0;     // 0 = use header value; else must be a multiple of 128 and <= header value
    int  matvec_rpb = 0;      // rows (warps) per block for opt matvec; 0 = kDefaultRpb[wtype]; else one of 1,2,4,8
};

void validate(const EngineOptions&, WeightType);  // throws

}  // namespace cudallm
