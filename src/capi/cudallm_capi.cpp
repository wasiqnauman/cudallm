#include "cudallm/cudallm.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <string>

#include "common/cuda_check.h"
#include "engine/engine.h"
#include "kernels/kernels.h"

struct cudallm_engine {
    cudallm::Engine* impl;
};

namespace {

thread_local std::string g_last_error;

void set_error(const char* msg) { g_last_error = msg ? msg : ""; }

// Maps a caught exception to the C error code (DESIGN.md §11).
int error_code(const std::exception& ex) {
    if (dynamic_cast<const cudallm::ArgError*>(&ex)) return CUDALLM_ERR_INVALID_ARG;
    if (dynamic_cast<const cudallm::FormatError*>(&ex)) return CUDALLM_ERR_FORMAT;
    if (dynamic_cast<const cudallm::IoError*>(&ex)) return CUDALLM_ERR_IO;
    if (std::strstr(ex.what(), "CUDA")) return CUDALLM_ERR_CUDA;
    return CUDALLM_ERR_INTERNAL;
}

template <typename F>
int guarded(F&& f) {
    try {
        f();
        return CUDALLM_OK;
    } catch (const std::exception& ex) {
        set_error(ex.what());
        return error_code(ex);
    } catch (...) {
        set_error("unknown exception");
        return CUDALLM_ERR_INTERNAL;
    }
}

void require(bool cond, const char* msg) {
    if (!cond) throw cudallm::ArgError(msg);
}

// Device buffer owned for the duration of one test-hook call.
class DevBuf {
public:
    explicit DevBuf(size_t bytes) {
        if (bytes) CUDALLM_CUDA_CHECK(cudaMalloc(&p_, bytes));
    }
    ~DevBuf() {
        if (p_) cudaFree(p_);
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    template <typename T>
    T* as() const {
        return static_cast<T*>(p_);
    }
    void upload(const void* src, size_t bytes) { CUDALLM_CUDA_CHECK(cudaMemcpy(p_, src, bytes, cudaMemcpyHostToDevice)); }
    void download(void* dst, size_t bytes) const { CUDALLM_CUDA_CHECK(cudaMemcpy(dst, p_, bytes, cudaMemcpyDeviceToHost)); }

private:
    void* p_ = nullptr;
};

void sync_and_check() {
    CUDALLM_CUDA_CHECK(cudaGetLastError());
    CUDALLM_CUDA_CHECK(cudaDeviceSynchronize());
}

cudallm::WeightType to_weight_type(int code) {
    require(code == 1 || code == 3 || code == 4, "weight_dtype must be 1 (bf16), 3 (int8) or 4 (int4)");
    return static_cast<cudallm::WeightType>(code);
}

}  // namespace

extern "C" {

const char* cudallm_last_error(void) { return g_last_error.c_str(); }

void cudallm_default_options(cudallm_options* out) {
    if (!out) return;
    out->matvec_impl = 1;
    out->attn_impl = 1;
    out->fuse = 1;
    out->cuda_graph = 1;
    out->max_seq_len = 0;
    out->matvec_rpb = 0;
}

int cudallm_create(const char* path, const cudallm_options* opts, cudallm_engine** out) {
    return guarded([&] {
        require(path && opts && out, "cudallm_create: null argument");
        *out = nullptr;
        require(opts->matvec_impl == 0 || opts->matvec_impl == 1, "matvec_impl must be 0 or 1");
        require(opts->attn_impl == 0 || opts->attn_impl == 1, "attn_impl must be 0 or 1");
        cudallm::EngineOptions o;
        o.matvec = static_cast<cudallm::MatvecImpl>(opts->matvec_impl);
        o.attn = static_cast<cudallm::AttnImpl>(opts->attn_impl);
        o.fuse = opts->fuse != 0;
        o.cuda_graph = opts->cuda_graph != 0;
        o.max_seq_len = opts->max_seq_len;
        o.matvec_rpb = opts->matvec_rpb;
        auto* e = new cudallm_engine{nullptr};
        try {
            e->impl = new cudallm::Engine(path, o);
        } catch (...) {
            delete e;
            throw;
        }
        *out = e;
    });
}

void cudallm_destroy(cudallm_engine* e) {
    if (!e) return;
    delete e->impl;
    delete e;
}

int cudallm_get_config(cudallm_engine* e, cudallm_config_info* out) {
    return guarded([&] {
        require(e && e->impl && out, "cudallm_get_config: null argument");
        const cudallm::ModelConfig& c = e->impl->config();
        out->weight_dtype = static_cast<int>(c.wtype);
        out->hidden = c.hidden;
        out->inter = c.inter;
        out->n_layers = c.n_layers;
        out->n_heads = c.n_heads;
        out->n_kv_heads = c.n_kv_heads;
        out->head_dim = c.head_dim;
        out->vocab = c.vocab;
        out->max_seq_len = e->impl->options().max_seq_len ? e->impl->options().max_seq_len : c.max_seq_len;
        out->group_size = c.group_size;
        out->rms_eps = c.rms_eps;
        out->weight_bytes = e->impl->weight_bytes();
    });
}

int cudallm_reset(cudallm_engine* e) {
    return guarded([&] {
        require(e && e->impl, "cudallm_reset: null engine");
        e->impl->reset();
    });
}

int cudallm_pos(cudallm_engine* e) {
    if (!e || !e->impl) {
        set_error("cudallm_pos: null engine");
        return -1;
    }
    return e->impl->pos();
}

int cudallm_step(cudallm_engine* e, int token, float* logits_out, int* next_token_out) {
    return guarded([&] {
        require(e && e->impl, "cudallm_step: null engine");
        require(token >= -1, "cudallm_step: token must be >= -1");
        e->impl->step(token, logits_out, next_token_out);
    });
}

int cudallm_step_debug(cudallm_engine* e, int token, float* residuals_out, float* logits_out, int* next_token_out) {
    return guarded([&] {
        require(e && e->impl, "cudallm_step_debug: null engine");
        require(token >= -1, "cudallm_step_debug: token must be >= -1");
        require(residuals_out, "cudallm_step_debug: residuals_out is null");
        e->impl->step_debug(token, residuals_out, logits_out, next_token_out);
    });
}

int cudallm_kernel_matvec(int impl, int weight_dtype, int epilogue, int fuse_norm, int rpb, const void* w,
                          unsigned long long w_bytes, const void* scales, unsigned long long scales_bytes,
                          const float* x, const float* norm_w, float eps, float* y, int rows, int cols) {
    return guarded([&] {
        require(w && x && y && w_bytes > 0, "cudallm_kernel_matvec: null argument");
        require(rows > 0 && cols > 0, "cudallm_kernel_matvec: bad shape");
        require(impl == 0 || impl == 1, "cudallm_kernel_matvec: impl must be 0 or 1");
        require(epilogue >= 0 && epilogue <= 2, "cudallm_kernel_matvec: bad epilogue");
        const cudallm::WeightType wt = to_weight_type(weight_dtype);
        const auto ep = static_cast<cudallm::Epilogue>(epilogue);
        if (impl == 0) {
            require(weight_dtype == 1 && epilogue == 0 && fuse_norm == 0,
                    "cudallm_kernel_matvec: naive requires bf16, STORE and no fused norm");
        }
        if (fuse_norm) require(norm_w != nullptr, "cudallm_kernel_matvec: fuse_norm needs norm_w");
        if (wt != cudallm::WeightType::BF16) require(scales && scales_bytes > 0, "cudallm_kernel_matvec: quantized weights need scales");
        const int out_rows = (ep == cudallm::Epilogue::SWIGLU) ? rows / 2 : rows;

        DevBuf dw(w_bytes), ds(scales ? scales_bytes : 0), dx(sizeof(float) * cols),
            dn(fuse_norm ? sizeof(float) * cols : 0), dy(sizeof(float) * out_rows);
        dw.upload(w, w_bytes);
        if (scales) ds.upload(scales, scales_bytes);
        dx.upload(x, sizeof(float) * cols);
        if (fuse_norm) dn.upload(norm_w, sizeof(float) * cols);
        dy.upload(y, sizeof(float) * out_rows);

        cudallm::MatvecArgs a{};
        a.w = dw.as<void>();
        a.scales = scales ? ds.as<void>() : nullptr;
        a.x = dx.as<float>();
        a.norm_w = fuse_norm ? dn.as<float>() : nullptr;
        a.eps = eps;
        a.y = dy.as<float>();
        a.rows = rows;
        a.cols = cols;
        if (impl == 0) {
            cudallm::launch_matvec_naive(a, 0);
        } else {
            cudallm::launch_matvec_opt(wt, ep, fuse_norm != 0, a, rpb ? rpb : cudallm::default_rpb(wt), 0);
        }
        sync_and_check();
        dy.download(y, sizeof(float) * out_rows);
    });
}

int cudallm_kernel_rmsnorm(const float* x, const float* w, float eps, float* y, int n) {
    return guarded([&] {
        require(x && w && y && n > 0, "cudallm_kernel_rmsnorm: bad argument");
        const size_t bytes = sizeof(float) * n;
        DevBuf dx(bytes), dw(bytes), dy(bytes);
        dx.upload(x, bytes);
        dw.upload(w, bytes);
        cudallm::launch_rmsnorm(dx.as<float>(), dw.as<float>(), eps, dy.as<float>(), n, 0);
        sync_and_check();
        dy.download(y, bytes);
    });
}

int cudallm_kernel_rope_kv(int fused, float* qkv, const float* inv_freq, int pos, unsigned short* k_cache,
                           unsigned short* v_cache, int nh, int nkv, int hd, int max_seq) {
    return guarded([&] {
        require(qkv && inv_freq && k_cache && v_cache, "cudallm_kernel_rope_kv: null argument");
        require(nh > 0 && nkv > 0 && hd > 0 && max_seq > 0 && pos >= 0 && pos < max_seq, "cudallm_kernel_rope_kv: bad dimensions");
        const size_t qkv_bytes = sizeof(float) * (nh + 2 * nkv) * hd;
        const size_t cache_bytes = sizeof(__half) * static_cast<size_t>(nkv) * max_seq * hd;
        DevBuf dqkv(qkv_bytes), dinv(sizeof(float) * hd / 2), dk(cache_bytes), dv(cache_bytes), dpos(sizeof(int));
        dqkv.upload(qkv, qkv_bytes);
        dinv.upload(inv_freq, sizeof(float) * hd / 2);
        dk.upload(k_cache, cache_bytes);
        dv.upload(v_cache, cache_bytes);
        dpos.upload(&pos, sizeof(int));
        if (fused) {
            cudallm::launch_rope_kv_fused(dqkv.as<float>(), dinv.as<float>(), dk.as<__half>(), dv.as<__half>(), dpos.as<int>(),
                                          nh, nkv, hd, max_seq, 0);
        } else {
            cudallm::launch_rope_inplace(dqkv.as<float>(), dinv.as<float>(), dpos.as<int>(), nh, nkv, hd, 0);
            cudallm::launch_kv_write(dqkv.as<float>(), dk.as<__half>(), dv.as<__half>(), dpos.as<int>(), nh, nkv, hd, max_seq, 0);
        }
        sync_and_check();
        dqkv.download(qkv, qkv_bytes);
        dk.download(k_cache, cache_bytes);
        dv.download(v_cache, cache_bytes);
    });
}

int cudallm_kernel_attention(int impl, const float* q, const unsigned short* k_cache, const unsigned short* v_cache,
                             int ctx, float* out, int nh, int nkv, int hd, int max_seq) {
    return guarded([&] {
        require(q && k_cache && v_cache && out, "cudallm_kernel_attention: null argument");
        require(impl == 0 || impl == 1, "cudallm_kernel_attention: impl must be 0 or 1");
        require(nh > 0 && nkv > 0 && hd > 0 && max_seq > 0 && ctx >= 1 && ctx <= max_seq, "cudallm_kernel_attention: bad dimensions");
        const size_t q_bytes = sizeof(float) * nh * hd;
        const size_t cache_bytes = sizeof(__half) * static_cast<size_t>(nkv) * max_seq * hd;
        const int pos = ctx - 1;
        const int n_split = (max_seq + cudallm::kAttnChunk - 1) / cudallm::kAttnChunk;
        DevBuf dq(q_bytes), dk(cache_bytes), dv(cache_bytes), dpos(sizeof(int)), dout(q_bytes),
            dpo(impl == 1 ? sizeof(float) * nh * n_split * hd : 0), dpml(impl == 1 ? sizeof(float) * nh * n_split * 2 : 0);
        dq.upload(q, q_bytes);
        dk.upload(k_cache, cache_bytes);
        dv.upload(v_cache, cache_bytes);
        dpos.upload(&pos, sizeof(int));
        if (impl == 0) {
            cudallm::launch_attention_naive(dq.as<float>(), dk.as<__half>(), dv.as<__half>(), dpos.as<int>(), dout.as<float>(), nh,
                                            nkv, hd, max_seq, 0);
        } else {
            cudallm::launch_attention_opt(dq.as<float>(), dk.as<__half>(), dv.as<__half>(), dpos.as<int>(), dpo.as<float>(),
                                          dpml.as<float>(), dout.as<float>(), nh, nkv, hd, max_seq, 0);
        }
        sync_and_check();
        dout.download(out, q_bytes);
    });
}

int cudallm_kernel_silu_mul(const float* gu, float* act, int inter) {
    return guarded([&] {
        require(gu && act && inter > 0, "cudallm_kernel_silu_mul: bad argument");
        DevBuf dgu(sizeof(float) * 2 * inter), dact(sizeof(float) * inter);
        dgu.upload(gu, sizeof(float) * 2 * inter);
        cudallm::launch_silu_mul(dgu.as<float>(), dact.as<float>(), inter, 0);
        sync_and_check();
        dact.download(act, sizeof(float) * inter);
    });
}

int cudallm_kernel_argmax(const float* logits, int n, int* out_idx) {
    return guarded([&] {
        require(logits && out_idx && n > 0, "cudallm_kernel_argmax: bad argument");
        DevBuf dl(sizeof(float) * n), dpv(sizeof(float) * cudallm::kArgmaxBlocks), dpi(sizeof(int) * cudallm::kArgmaxBlocks),
            dtok(sizeof(int)), dpos(sizeof(int));
        dl.upload(logits, sizeof(float) * n);
        const int zero = 0;
        dtok.upload(&zero, sizeof(int));
        dpos.upload(&zero, sizeof(int));
        cudallm::launch_argmax(dl.as<float>(), n, dpv.as<float>(), dpi.as<int>(), dtok.as<int>(), dpos.as<int>(), 0);
        sync_and_check();
        dtok.download(out_idx, sizeof(int));
    });
}

}  // extern "C"
