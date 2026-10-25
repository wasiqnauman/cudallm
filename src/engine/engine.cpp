#include "engine/engine.h"

#include <cstdio>
#include <string>

#include "common/cuda_check.h"

namespace cudallm {

void validate(const EngineOptions& o, WeightType wtype) {
    if (o.matvec != MatvecImpl::NAIVE && o.matvec != MatvecImpl::OPT) throw ArgError("invalid matvec implementation");
    if (o.attn != AttnImpl::NAIVE && o.attn != AttnImpl::OPT) throw ArgError("invalid attention implementation");
    if (o.matvec == MatvecImpl::NAIVE) {
        if (wtype != WeightType::BF16) throw ArgError("naive matvec requires bf16 weights");
        if (o.fuse) throw ArgError("naive matvec requires fuse=false");
    }
    if (o.matvec_rpb != 0 && o.matvec_rpb != 1 && o.matvec_rpb != 2 && o.matvec_rpb != 4 && o.matvec_rpb != 8) {
        throw ArgError("matvec_rpb must be one of 0, 1, 2, 4, 8");
    }
    if (o.max_seq_len < 0 || o.max_seq_len % 128 != 0) throw ArgError("max_seq_len must be 0 or a positive multiple of 128");
}

namespace {

template <typename T>
void alloc(T** p, size_t count) {
    CUDALLM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(p), count * sizeof(T)));
}

}  // namespace

Engine::Engine(const std::string& path, const EngineOptions& opts) : opts_(opts) {
    weights_ = std::make_unique<Weights>(path);
    const ModelConfig& c = weights_->config();
    validate(opts_, c.wtype);
    if (opts_.max_seq_len > c.max_seq_len) throw ArgError("max_seq_len exceeds the file's max_seq_len");
    max_seq_ = opts_.max_seq_len ? opts_.max_seq_len : c.max_seq_len;

    try {
        CUDALLM_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        const size_t H = c.hidden, I = c.inter, V = c.vocab, S = max_seq_, KV = c.n_kv_heads, D = c.head_dim,
                     NH = c.n_heads;
        alloc(&x_, H);
        alloc(&xb_, H);
        alloc(&qkv_, (NH + 2 * KV) * D);
        alloc(&att_, NH * D);
        alloc(&tmp_, H);
        alloc(&gu_, 2 * I);
        alloc(&act_, I);
        alloc(&logits_, V);
        alloc(&k_cache_, static_cast<size_t>(c.n_layers) * KV * S * D);
        alloc(&v_cache_, static_cast<size_t>(c.n_layers) * KV * S * D);
        alloc(&attn_part_o_, NH * (S / kAttnChunk) * D);
        alloc(&attn_part_ml_, NH * (S / kAttnChunk) * 2);
        alloc(&argmax_part_v_, kArgmaxBlocks);
        alloc(&argmax_part_i_, kArgmaxBlocks);
        alloc(&d_token_, 1);
        alloc(&d_pos_, 1);
        CUDALLM_CUDA_CHECK(cudaMemset(d_token_, 0, sizeof(int)));
        CUDALLM_CUDA_CHECK(cudaMemset(d_pos_, 0, sizeof(int)));
        CUDALLM_CUDA_CHECK(cudaDeviceSynchronize());
    } catch (...) {
        free_all();
        throw;
    }
}

void Engine::free_all() noexcept {
    auto report = [](cudaError_t err, const char* what) {
        if (err != cudaSuccess) std::fprintf(stderr, "cudallm: %s failed in ~Engine: %s\n", what, cudaGetErrorString(err));
    };
    if (graph_exec_) report(cudaGraphExecDestroy(graph_exec_), "cudaGraphExecDestroy");
    if (graph_) report(cudaGraphDestroy(graph_), "cudaGraphDestroy");
    graph_exec_ = nullptr;
    graph_ = nullptr;
    void* bufs[] = {x_, xb_, qkv_, att_, tmp_, gu_, act_, logits_, k_cache_, v_cache_, attn_part_o_,
                    attn_part_ml_, argmax_part_v_, argmax_part_i_, d_token_, d_pos_};
    for (void* b : bufs) {
        if (b) report(cudaFree(b), "cudaFree");
    }
    x_ = xb_ = qkv_ = att_ = tmp_ = gu_ = act_ = logits_ = attn_part_o_ = attn_part_ml_ = argmax_part_v_ = nullptr;
    k_cache_ = v_cache_ = nullptr;
    argmax_part_i_ = d_token_ = d_pos_ = nullptr;
    if (stream_) report(cudaStreamDestroy(stream_), "cudaStreamDestroy");
    stream_ = nullptr;
}

Engine::~Engine() {
    try {
        if (stream_) cudaStreamSynchronize(stream_);
        free_all();
        weights_.reset();
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "cudallm: exception in ~Engine: %s\n", ex.what());
    } catch (...) {
        std::fprintf(stderr, "cudallm: unknown exception in ~Engine\n");
    }
}

const ModelConfig& Engine::config() const { return weights_->config(); }

const EngineOptions& Engine::options() const { return opts_; }

uint64_t Engine::weight_bytes() const { return weights_->lm_and_layer_bytes(); }

int Engine::pos() const { return host_pos_; }

void Engine::reset() {
    CUDALLM_CUDA_CHECK(cudaStreamSynchronize(stream_));
    CUDALLM_CUDA_CHECK(cudaMemsetAsync(d_pos_, 0, sizeof(int), stream_));
    CUDALLM_CUDA_CHECK(cudaStreamSynchronize(stream_));
    host_pos_ = 0;
}

void Engine::synchronize() { CUDALLM_CUDA_CHECK(cudaStreamSynchronize(stream_)); }

void Engine::step(int token, float* logits_out, int* next_token_out) {
    CUDALLM_CHECK(host_pos_ < max_seq_, "step: sequence is full (pos == max_seq_len); call reset()");
    if (token < -1 || token >= config().vocab) throw ArgError("step: token out of range");
    if (token >= 0) launch_set_token(d_token_, token, stream_);
    if (opts_.cuda_graph) {
        if (!graph_exec_) {
            CUDALLM_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
            try {
                enqueue_forward(false, nullptr);
            } catch (...) {
                cudaGraph_t broken = nullptr;
                cudaStreamEndCapture(stream_, &broken);
                if (broken) cudaGraphDestroy(broken);
                throw;
            }
            CUDALLM_CUDA_CHECK(cudaStreamEndCapture(stream_, &graph_));
            CUDALLM_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, 0));
        }
        CUDALLM_CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_));
    } else {
        enqueue_forward(false, nullptr);
    }
    host_pos_ += 1;
    if (logits_out || next_token_out) {
        CUDALLM_CUDA_CHECK(cudaStreamSynchronize(stream_));
        if (logits_out) {
            CUDALLM_CUDA_CHECK(cudaMemcpy(logits_out, logits_, sizeof(float) * config().vocab, cudaMemcpyDeviceToHost));
        }
        if (next_token_out) CUDALLM_CUDA_CHECK(cudaMemcpy(next_token_out, d_token_, sizeof(int), cudaMemcpyDeviceToHost));
    }
}

void Engine::step_debug(int token, float* residuals_out, float* logits_out, int* next_token_out) {
    CUDALLM_CHECK(host_pos_ < max_seq_, "step_debug: sequence is full (pos == max_seq_len); call reset()");
    if (token < -1 || token >= config().vocab) throw ArgError("step_debug: token out of range");
    if (!residuals_out) throw ArgError("step_debug: residuals_out is null");
    if (token >= 0) launch_set_token(d_token_, token, stream_);
    enqueue_forward(true, residuals_out);
    host_pos_ += 1;
    CUDALLM_CUDA_CHECK(cudaStreamSynchronize(stream_));
    if (logits_out) CUDALLM_CUDA_CHECK(cudaMemcpy(logits_out, logits_, sizeof(float) * config().vocab, cudaMemcpyDeviceToHost));
    if (next_token_out) CUDALLM_CUDA_CHECK(cudaMemcpy(next_token_out, d_token_, sizeof(int), cudaMemcpyDeviceToHost));
}

void Engine::launch_linear(const Linear& L, const float* x, float* y, Epilogue ep, const float* norm_w) {
    CUDALLM_CHECK(static_cast<const void*>(x) != static_cast<const void*>(y), "launch_linear: input and output alias");
    MatvecArgs a{};
    a.w = L.w;
    a.scales = L.scales;
    a.x = x;
    a.norm_w = norm_w;
    a.eps = config().rms_eps;
    a.y = y;
    a.rows = L.rows;
    a.cols = L.cols;
    if (opts_.matvec == MatvecImpl::NAIVE) {
        CUDALLM_CHECK(ep == Epilogue::STORE && norm_w == nullptr, "launch_linear: naive matvec supports STORE only, no fused norm");
        launch_matvec_naive(a, stream_);
    } else {
        const int rpb = opts_.matvec_rpb ? opts_.matvec_rpb : default_rpb(L.wtype);
        launch_matvec_opt(L.wtype, ep, norm_w != nullptr, a, rpb, stream_);
    }
}

void Engine::enqueue_forward(bool debug, float* residuals_out) {
    const ModelConfig& c = config();
    const Weights& W = *weights_;
    const int H = c.hidden, NH = c.n_heads, KV = c.n_kv_heads, D = c.head_dim;
    const size_t layer_cache = static_cast<size_t>(KV) * max_seq_ * D;
    auto snapshot = [&](int k) {
        if (!debug) return;
        CUDALLM_CUDA_CHECK(cudaStreamSynchronize(stream_));
        CUDALLM_CUDA_CHECK(cudaMemcpy(residuals_out + static_cast<size_t>(k) * H, x_, sizeof(float) * H, cudaMemcpyDeviceToHost));
    };
    auto attention = [&](int l) {
        const __half* kl = k_cache_ + l * layer_cache;
        const __half* vl = v_cache_ + l * layer_cache;
        if (opts_.attn == AttnImpl::NAIVE) {
            launch_attention_naive(qkv_, kl, vl, d_pos_, att_, NH, KV, D, max_seq_, stream_);
        } else {
            launch_attention_opt(qkv_, kl, vl, d_pos_, attn_part_o_, attn_part_ml_, att_, NH, KV, D, max_seq_, stream_);
        }
    };

    launch_embedding(W.tok_embeddings, d_token_, x_, H, c.group_size, stream_);
    snapshot(0);
    for (int l = 0; l < c.n_layers; ++l) {
        const LayerWeights& L = W.layers[l];
        __half* kl = k_cache_ + l * layer_cache;
        __half* vl = v_cache_ + l * layer_cache;
        if (!opts_.fuse) {
            launch_rmsnorm(x_, L.attn_norm, c.rms_eps, xb_, H, stream_);
            launch_linear(L.wqkv, xb_, qkv_, Epilogue::STORE, nullptr);
            launch_rope_inplace(qkv_, W.rope_inv_freq, d_pos_, NH, KV, D, stream_);
            launch_kv_write(qkv_, kl, vl, d_pos_, NH, KV, D, max_seq_, stream_);
            attention(l);
            launch_linear(L.wo, att_, tmp_, Epilogue::STORE, nullptr);
            launch_add_inplace(x_, tmp_, H, stream_);
            launch_rmsnorm(x_, L.ffn_norm, c.rms_eps, xb_, H, stream_);
            launch_linear(L.w_gate_up, xb_, gu_, Epilogue::STORE, nullptr);
            launch_silu_mul(gu_, act_, c.inter, stream_);
            launch_linear(L.w_down, act_, tmp_, Epilogue::STORE, nullptr);
            launch_add_inplace(x_, tmp_, H, stream_);
        } else {
            launch_linear(L.wqkv, x_, qkv_, Epilogue::STORE, L.attn_norm);
            launch_rope_kv_fused(qkv_, W.rope_inv_freq, kl, vl, d_pos_, NH, KV, D, max_seq_, stream_);
            attention(l);
            launch_linear(L.wo, att_, x_, Epilogue::ADD, nullptr);
            launch_linear(L.w_gate_up, x_, act_, Epilogue::SWIGLU, L.ffn_norm);
            launch_linear(L.w_down, act_, x_, Epilogue::ADD, nullptr);
        }
        snapshot(l + 1);
    }
    if (!opts_.fuse) {
        launch_rmsnorm(x_, W.final_norm, c.rms_eps, xb_, H, stream_);
        launch_linear(W.tok_embeddings, xb_, logits_, Epilogue::STORE, nullptr);
    } else {
        launch_linear(W.tok_embeddings, x_, logits_, Epilogue::STORE, W.final_norm);
    }
    launch_argmax(logits_, c.vocab, argmax_part_v_, argmax_part_i_, d_token_, d_pos_, stream_);
}

}  // namespace cudallm
