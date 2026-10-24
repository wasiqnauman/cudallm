#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/dtype.h"

namespace cudallm {

struct Linear {            // one matrix-vector operand
    const void* w;         // device ptr
    const void* scales;    // device ptr or nullptr (bf16)
    int rows, cols;
    WeightType wtype;
};

struct LayerWeights {
    const float* attn_norm;  Linear wqkv;  Linear wo;
    const float* ffn_norm;   Linear w_gate_up;  Linear w_down;
};

struct ModelConfig {
    WeightType wtype; int hidden, inter, n_layers, n_heads, n_kv_heads, head_dim,
    vocab, max_seq_len, group_size; float rms_eps;
};

class Weights {
public:
    explicit Weights(const std::string& path);   // throws on any error
    ~Weights();                                   // cudaFree
    Weights(const Weights&) = delete;
    Weights& operator=(const Weights&) = delete;
    const ModelConfig& config() const;
    const float* rope_inv_freq = nullptr; const float* final_norm = nullptr;
    Linear tok_embeddings{};        // also used as lm_head
    std::vector<LayerWeights> layers;
    uint64_t lm_and_layer_bytes() const; // bytes read per decode step = sum of nbytes of all tensors, each counted once

private:
    ModelConfig config_{};
    void* base_ = nullptr;
    uint64_t total_bytes_ = 0;
};

}  // namespace cudallm
