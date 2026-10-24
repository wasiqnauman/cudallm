#ifndef CUDALLM_CUDALLM_H
#define CUDALLM_CUDALLM_H

#ifdef __cplusplus
extern "C" {
#endif

#define CUDALLM_OK 0
#define CUDALLM_ERR_INVALID_ARG 1
#define CUDALLM_ERR_IO 2
#define CUDALLM_ERR_FORMAT 3
#define CUDALLM_ERR_CUDA 4
#define CUDALLM_ERR_STATE 5
#define CUDALLM_ERR_INTERNAL 6

typedef struct cudallm_engine cudallm_engine;

typedef struct {
    int matvec_impl;   /* 0 naive, 1 opt */
    int attn_impl;     /* 0 naive, 1 opt */
    int fuse;          /* 0/1 */
    int cuda_graph;    /* 0/1 */
    int max_seq_len;   /* 0 = from file */
    int matvec_rpb;    /* 0 = default */
} cudallm_options;

typedef struct {
    int weight_dtype;  /* 1 bf16, 3 int8, 4 int4 */
    int hidden, inter, n_layers, n_heads, n_kv_heads, head_dim, vocab, max_seq_len, group_size;
    float rms_eps;
    unsigned long long weight_bytes;
} cudallm_config_info;

const char* cudallm_last_error(void);                 /* thread-local message of the last failure */
void cudallm_default_options(cudallm_options* out);        /* opt, opt, 1, 1, 0, 0 */
int  cudallm_create(const char* path, const cudallm_options* opts, cudallm_engine** out);
void cudallm_destroy(cudallm_engine* e);
int  cudallm_get_config(cudallm_engine* e, cudallm_config_info* out);
int  cudallm_reset(cudallm_engine* e);
int  cudallm_pos(cudallm_engine* e);
int  cudallm_step(cudallm_engine* e, int token, float* logits_out, int* next_token_out);
int  cudallm_step_debug(cudallm_engine* e, int token, float* residuals_out, float* logits_out, int* next_token_out);

/* ---- single-kernel test hooks: host pointers in/out; allocate, copy, run, copy back, free ---- */
int cudallm_kernel_matvec(int impl, int weight_dtype, int epilogue, int fuse_norm, int rpb,
                     const void* w, unsigned long long w_bytes, const void* scales, unsigned long long scales_bytes,
                     const float* x, const float* norm_w, float eps,
                     float* y /* in/out (ADD reads it) */, int rows, int cols);
int cudallm_kernel_rmsnorm(const float* x, const float* w, float eps, float* y, int n);
int cudallm_kernel_rope_kv(int fused, float* qkv /* in/out */, const float* inv_freq, int pos,
                      unsigned short* k_cache, unsigned short* v_cache /* in/out, fp16 bits, [nkv][max_seq][hd] */,
                      int nh, int nkv, int hd, int max_seq);
int cudallm_kernel_attention(int impl, const float* q, const unsigned short* k_cache, const unsigned short* v_cache,
                        int ctx, float* out, int nh, int nkv, int hd, int max_seq);
int cudallm_kernel_silu_mul(const float* gu, float* act, int inter);
int cudallm_kernel_argmax(const float* logits, int n, int* out_idx);

#ifdef __cplusplus
}
#endif

#endif /* CUDALLM_CUDALLM_H */
