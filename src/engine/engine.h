#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>

#include "engine/options.h"
#include "kernels/kernels.h"
#include "model/weights.h"

namespace cudallm {

class Engine {
public:
    Engine(const std::string& path, const EngineOptions& opts);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    const ModelConfig& config() const;
    const EngineOptions& options() const;
    uint64_t weight_bytes() const;
    int  pos() const;                       // host mirror of the device position
    void reset();                           // sync, d_pos = 0, host_pos = 0
    // Feed one token. token >= 0: use it. token == -1: use the previous step's argmax (already on device).
    // logits_out (host, vocab floats) and next_token_out may be null; if both are null the call does not synchronize.
    void step(int token, float* logits_out, int* next_token_out);
    // Same as step() but never uses the CUDA graph and copies the residual stream after the
    // embedding and after each layer into residuals_out [(n_layers+1) * hidden] (host). logits_out may be null.
    void step_debug(int token, float* residuals_out, float* logits_out, int* next_token_out);
    void synchronize();

private:
    void enqueue_forward(bool debug, float* residuals_out); // enqueues one full decode step on stream_
    void launch_linear(const Linear& L, const float* x, float* y, Epilogue ep, const float* norm_w);

    std::unique_ptr<Weights> weights_;
    EngineOptions opts_;
    int host_pos_ = 0;
};

}  // namespace cudallm
