#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "common/dtype.h"
#include "model/weights.h"

namespace cudallm {

enum class Epilogue : int { STORE = 0, ADD = 1, SWIGLU = 2 };
constexpr int kAttnChunk = 128;
constexpr int kArgmaxBlocks = 256;
// Default rows-per-block for opt matvec, indexed by WeightType. Set from the Phase 3 sweep winners
// (results/sweep_matvec.json).
constexpr int kDefaultRpbBf16 = 8;
constexpr int kDefaultRpbInt8 = 8;
constexpr int kDefaultRpbInt4 = 8;
int default_rpb(WeightType w);

struct MatvecArgs {
    const void* w; const void* scales; const float* x; const float* norm_w; float eps;
    float* y; int rows; int cols;
};

void launch_embedding(const Linear& emb, const int* d_token, float* x, int hidden, int group_size, cudaStream_t);
void launch_rmsnorm(const float* x, const float* w, float eps, float* y, int n, cudaStream_t);
void launch_matvec_naive(const MatvecArgs&, cudaStream_t);                  // bf16, STORE only
void launch_matvec_opt(WeightType, Epilogue, bool fuse_norm, const MatvecArgs&, int rpb, cudaStream_t);
void launch_rope_inplace(float* qkv, const float* inv_freq, const int* d_pos, int nh, int nkv, int hd, cudaStream_t);
void launch_kv_write(const float* qkv, __half* k_cache_l, __half* v_cache_l, const int* d_pos, int nh, int nkv, int hd, int max_seq, cudaStream_t);
void launch_rope_kv_fused(float* qkv, const float* inv_freq, __half* k_cache_l, __half* v_cache_l, const int* d_pos, int nh, int nkv, int hd, int max_seq, cudaStream_t);
void launch_attention_naive(const float* q, const __half* k_l, const __half* v_l, const int* d_pos, float* out, int nh, int nkv, int hd, int max_seq, cudaStream_t);
void launch_attention_opt(const float* q, const __half* k_l, const __half* v_l, const int* d_pos, float* part_o, float* part_ml, float* out, int nh, int nkv, int hd, int max_seq, cudaStream_t);
void launch_add_inplace(float* x, const float* y, int n, cudaStream_t);
void launch_silu_mul(const float* gu, float* act, int inter, cudaStream_t);
void launch_set_token(int* d_token, int token, cudaStream_t);
void launch_argmax(const float* logits, int n, float* part_v, int* part_i, int* d_token, int* d_pos, cudaStream_t);
// bandwidth.cu
void launch_bw_read(const uint4* src, size_t n_uint4, unsigned* sink, int grid, cudaStream_t);
void launch_bw_write(uint4* dst, size_t n_uint4, int grid, cudaStream_t);
void launch_bw_copy(const uint4* src, uint4* dst, size_t n_uint4, int grid, cudaStream_t);

}  // namespace cudallm
