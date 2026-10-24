#include "engine/engine.h"

#include <stdexcept>

#include "common/cuda_check.h"

namespace cudallm {

void validate(const EngineOptions&, WeightType) { throw std::runtime_error("not implemented: validate(EngineOptions)"); }

Engine::Engine(const std::string&, const EngineOptions& opts) : opts_(opts) {
    throw std::runtime_error("not implemented: Engine");
}

Engine::~Engine() {}

const ModelConfig& Engine::config() const { return weights_->config(); }

const EngineOptions& Engine::options() const { return opts_; }

uint64_t Engine::weight_bytes() const { return weights_->lm_and_layer_bytes(); }

int Engine::pos() const { return host_pos_; }

void Engine::reset() { throw std::runtime_error("not implemented: Engine::reset"); }

void Engine::step(int, float*, int*) { throw std::runtime_error("not implemented: Engine::step"); }

void Engine::step_debug(int, float*, float*, int*) { throw std::runtime_error("not implemented: Engine::step_debug"); }

void Engine::synchronize() { throw std::runtime_error("not implemented: Engine::synchronize"); }

void Engine::enqueue_forward(bool, float*) { throw std::runtime_error("not implemented: Engine::enqueue_forward"); }

void Engine::launch_linear(const Linear&, const float*, float*, Epilogue, const float*) {
    throw std::runtime_error("not implemented: Engine::launch_linear");
}

}  // namespace cudallm
