#pragma once

#include <cstddef>
#include <cstdint>

namespace cudallm {

constexpr char kCudallmMagic[5] = "CLLM";
constexpr uint32_t kCudallmVersion = 1;

#pragma pack(push, 1)
struct CudallmHeader {
    char magic[4];
    uint32_t version;
    uint32_t weight_dtype;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t vocab_size;
    uint32_t max_seq_len;
    float rms_eps;
    float rope_theta;
    uint32_t group_size;
    uint32_t n_tensors;
    uint32_t reserved0;
    uint64_t table_offset;
    uint8_t zero[184];
};

struct CudallmTensorEntry {
    char name[48];
    uint32_t dtype;
    uint32_t rows;
    uint32_t cols;
    uint32_t reserved;
    uint64_t offset;
    uint64_t nbytes;
};
#pragma pack(pop)

static_assert(sizeof(CudallmHeader) == 256, "CudallmHeader must be 256 bytes");
static_assert(offsetof(CudallmHeader, table_offset) == 64, "table_offset must be at byte 64");
static_assert(sizeof(CudallmTensorEntry) == 80, "CudallmTensorEntry must be 80 bytes");

}  // namespace cudallm
