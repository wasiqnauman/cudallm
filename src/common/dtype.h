#pragma once

#include <cstdint>

namespace cudallm {

enum class DType : uint32_t { F32 = 0, BF16 = 1, F16 = 2, I8 = 3, I4X2 = 4 };
enum class WeightType : uint32_t { BF16 = 1, INT8 = 3, INT4 = 4 };

inline const char* to_string(WeightType w) {
    switch (w) {
        case WeightType::BF16: return "bf16";
        case WeightType::INT8: return "int8";
        case WeightType::INT4: return "int4";
    }
    return "unknown";
}

}  // namespace cudallm
