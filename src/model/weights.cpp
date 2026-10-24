#include "model/weights.h"

#include <stdexcept>

namespace cudallm {

Weights::Weights(const std::string& /*path*/) { throw std::runtime_error("not implemented: Weights loader"); }

Weights::~Weights() {}

const ModelConfig& Weights::config() const { return config_; }

uint64_t Weights::lm_and_layer_bytes() const { return total_bytes_; }

}  // namespace cudallm
