#include "cudallm/cudallm.h"

#include <string>

namespace {

thread_local std::string g_last_error;

int not_implemented(const char* what) {
    g_last_error = std::string("not implemented: ") + what;
    return CUDALLM_ERR_INTERNAL;
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

int cudallm_create(const char*, const cudallm_options*, cudallm_engine**) { return not_implemented("cudallm_create"); }
void cudallm_destroy(cudallm_engine*) {}
int cudallm_get_config(cudallm_engine*, cudallm_config_info*) { return not_implemented("cudallm_get_config"); }
int cudallm_reset(cudallm_engine*) { return not_implemented("cudallm_reset"); }
int cudallm_pos(cudallm_engine*) { return -1; }
int cudallm_step(cudallm_engine*, int, float*, int*) { return not_implemented("cudallm_step"); }
int cudallm_step_debug(cudallm_engine*, int, float*, float*, int*) { return not_implemented("cudallm_step_debug"); }

int cudallm_kernel_matvec(int, int, int, int, int, const void*, unsigned long long, const void*, unsigned long long,
                          const float*, const float*, float, float*, int, int) {
    return not_implemented("cudallm_kernel_matvec");
}
int cudallm_kernel_rmsnorm(const float*, const float*, float, float*, int) { return not_implemented("cudallm_kernel_rmsnorm"); }
int cudallm_kernel_rope_kv(int, float*, const float*, int, unsigned short*, unsigned short*, int, int, int, int) {
    return not_implemented("cudallm_kernel_rope_kv");
}
int cudallm_kernel_attention(int, const float*, const unsigned short*, const unsigned short*, int, float*, int, int, int, int) {
    return not_implemented("cudallm_kernel_attention");
}
int cudallm_kernel_silu_mul(const float*, float*, int) { return not_implemented("cudallm_kernel_silu_mul"); }
int cudallm_kernel_argmax(const float*, int, int*) { return not_implemented("cudallm_kernel_argmax"); }

}  // extern "C"
