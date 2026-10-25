#include "model/weights.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "common/cuda_check.h"
#include "model/cudallm_format.h"

namespace cudallm {

namespace {

// Llama-3.2-1B dimensions (DESIGN.md §4); the loader accepts nothing else.
constexpr uint32_t kHidden = 2048;
constexpr uint32_t kInter = 8192;
constexpr uint32_t kLayers = 16;
constexpr uint32_t kHeads = 32;
constexpr uint32_t kKvHeads = 8;
constexpr uint32_t kHeadDim = 64;
constexpr uint32_t kVocab = 128256;
constexpr uint32_t kInt4Group = 128;
constexpr uint64_t kAlign = 256;
constexpr size_t kChunkBytes = 64ull << 20;

struct ExpectedTensor {
    std::string name;
    DType dtype;
    uint32_t rows, cols;
};

uint64_t tensor_nbytes(DType dt, uint64_t rows, uint64_t cols) {
    switch (dt) {
        case DType::F32: return rows * cols * 4;
        case DType::BF16:
        case DType::F16: return rows * cols * 2;
        case DType::I8: return rows * cols;
        case DType::I4X2: return rows * cols / 2;
    }
    throw FormatError("unknown tensor dtype");
}

void add_linear(std::vector<ExpectedTensor>& out, const std::string& name, WeightType wt, uint32_t rows,
                uint32_t cols) {
    switch (wt) {
        case WeightType::BF16:
            out.push_back({name + ".weight", DType::BF16, rows, cols});
            break;
        case WeightType::INT8:
            out.push_back({name + ".weight", DType::I8, rows, cols});
            out.push_back({name + ".scales", DType::F32, rows, 1});
            break;
        case WeightType::INT4:
            out.push_back({name + ".weight", DType::I4X2, rows, cols});
            out.push_back({name + ".scales", DType::F16, rows, cols / kInt4Group});
            break;
    }
}

// The exact §7.5 tensor sequence for a given weight type.
std::vector<ExpectedTensor> expected_tensors(WeightType wt) {
    std::vector<ExpectedTensor> t;
    t.push_back({"rope.inv_freq", DType::F32, kHeadDim / 2, 1});
    add_linear(t, "tok_embeddings", wt, kVocab, kHidden);
    t.push_back({"norm.weight", DType::F32, kHidden, 1});
    for (uint32_t i = 0; i < kLayers; ++i) {
        const std::string p = "layers." + std::to_string(i) + ".";
        t.push_back({p + "attn_norm.weight", DType::F32, kHidden, 1});
        add_linear(t, p + "wqkv", wt, (kHeads + 2 * kKvHeads) * kHeadDim, kHidden);
        add_linear(t, p + "wo", wt, kHidden, kHidden);
        t.push_back({p + "ffn_norm.weight", DType::F32, kHidden, 1});
        add_linear(t, p + "w_gate_up", wt, 2 * kInter, kHidden);
        add_linear(t, p + "w_down", wt, kHidden, kInter);
    }
    return t;
}

void read_exact(std::ifstream& f, void* dst, size_t n, const std::string& path, const char* what) {
    f.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
    if (static_cast<size_t>(f.gcount()) != n) {
        throw IoError(path + ": short read (" + what + ")");
    }
}

void expect_eq(uint64_t got, uint64_t want, const char* field) {
    if (got != want) {
        throw FormatError(std::string("header field ") + field + " = " + std::to_string(got) + ", expected " +
                          std::to_string(want));
    }
}

}  // namespace

Weights::Weights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw IoError("cannot open " + path);
    f.seekg(0, std::ios::end);
    const uint64_t file_size = static_cast<uint64_t>(f.tellg());
    f.seekg(0, std::ios::beg);

    // Step 1: header.
    CudallmHeader h{};
    if (file_size < sizeof(h)) throw FormatError(path + ": file too small for header");
    read_exact(f, &h, sizeof(h), path, "header");
    if (std::memcmp(h.magic, kCudallmMagic, 4) != 0) throw FormatError(path + ": bad magic");
    expect_eq(h.version, kCudallmVersion, "version");
    if (h.weight_dtype != 1 && h.weight_dtype != 3 && h.weight_dtype != 4) {
        throw FormatError("unsupported weight_dtype " + std::to_string(h.weight_dtype));
    }
    const WeightType wt = static_cast<WeightType>(h.weight_dtype);
    expect_eq(h.hidden_size, kHidden, "hidden_size");
    expect_eq(h.intermediate_size, kInter, "intermediate_size");
    expect_eq(h.n_layers, kLayers, "n_layers");
    expect_eq(h.n_heads, kHeads, "n_heads");
    expect_eq(h.n_kv_heads, kKvHeads, "n_kv_heads");
    expect_eq(h.head_dim, kHeadDim, "head_dim");
    expect_eq(h.vocab_size, kVocab, "vocab_size");
    if (h.max_seq_len == 0 || h.max_seq_len % 128 != 0) {
        throw FormatError("max_seq_len must be a positive multiple of 128");
    }
    expect_eq(h.n_tensors, wt == WeightType::BF16 ? 99 : 164, "n_tensors");
    expect_eq(h.table_offset, 256, "table_offset");
    expect_eq(h.group_size, wt == WeightType::INT4 ? kInt4Group : 0, "group_size");

    // Step 2: tensor table.
    const std::vector<ExpectedTensor> expected = expected_tensors(wt);
    CUDALLM_CHECK(expected.size() == h.n_tensors, "internal: expected tensor list size mismatch");
    std::vector<CudallmTensorEntry> table(h.n_tensors);
    const uint64_t table_end = h.table_offset + uint64_t(h.n_tensors) * sizeof(CudallmTensorEntry);
    if (file_size < table_end) throw FormatError(path + ": file too small for tensor table");
    f.seekg(static_cast<std::streamoff>(h.table_offset));
    read_exact(f, table.data(), table.size() * sizeof(CudallmTensorEntry), path, "tensor table");

    uint64_t prev_end = table_end;
    for (size_t k = 0; k < table.size(); ++k) {
        const CudallmTensorEntry& e = table[k];
        const ExpectedTensor& x = expected[k];
        const std::string name(e.name, strnlen(e.name, sizeof(e.name)));
        if (name != x.name) throw FormatError("tensor " + std::to_string(k) + " is '" + name + "', expected '" + x.name + "'");
        if (e.dtype != static_cast<uint32_t>(x.dtype) || e.rows != x.rows || e.cols != x.cols) {
            throw FormatError("tensor '" + name + "' has wrong dtype or shape");
        }
        if (e.offset % kAlign != 0) throw FormatError("tensor '" + name + "' offset not 256-aligned");
        if (e.offset < prev_end) throw FormatError("tensor '" + name + "' offset overlaps previous data");
        if (e.nbytes != tensor_nbytes(x.dtype, x.rows, x.cols)) throw FormatError("tensor '" + name + "' has wrong nbytes");
        prev_end = e.offset + e.nbytes;
    }

    // Step 3: one device allocation, filled through a 64 MiB pinned staging buffer.
    const uint64_t data_begin = table.front().offset;
    const uint64_t data_end = table.back().offset + table.back().nbytes;
    if (file_size < data_end) throw FormatError(path + ": file truncated (data region)");
    total_bytes_ = 0;
    for (const auto& e : table) total_bytes_ += e.nbytes;

    CUDALLM_CUDA_CHECK(cudaMalloc(&base_, data_end - data_begin));
    void* pinned = nullptr;
    try {
        CUDALLM_CUDA_CHECK(cudaMallocHost(&pinned, kChunkBytes));
        f.seekg(static_cast<std::streamoff>(data_begin));
        for (uint64_t done = 0; done < data_end - data_begin;) {
            const size_t n = static_cast<size_t>(std::min<uint64_t>(kChunkBytes, data_end - data_begin - done));
            read_exact(f, pinned, n, path, "data region");
            CUDALLM_CUDA_CHECK(cudaMemcpy(static_cast<char*>(base_) + done, pinned, n, cudaMemcpyHostToDevice));
            done += n;
        }
        CUDALLM_CUDA_CHECK(cudaFreeHost(pinned));
        pinned = nullptr;
    } catch (...) {
        if (pinned) cudaFreeHost(pinned);
        cudaFree(base_);
        base_ = nullptr;
        throw;
    }

    // Step 4: pointers into the single allocation.
    auto ptr = [&](size_t k) -> const void* { return static_cast<const char*>(base_) + (table[k].offset - data_begin); };
    const bool quant = wt != WeightType::BF16;
    size_t k = 0;
    auto take_linear = [&](int rows, int cols) {
        Linear L{};
        L.w = ptr(k++);
        L.scales = quant ? ptr(k++) : nullptr;
        L.rows = rows;
        L.cols = cols;
        L.wtype = wt;
        return L;
    };
    rope_inv_freq = static_cast<const float*>(ptr(k++));
    tok_embeddings = take_linear(kVocab, kHidden);
    final_norm = static_cast<const float*>(ptr(k++));
    layers.resize(kLayers);
    for (auto& lw : layers) {
        lw.attn_norm = static_cast<const float*>(ptr(k++));
        lw.wqkv = take_linear((kHeads + 2 * kKvHeads) * kHeadDim, kHidden);
        lw.wo = take_linear(kHidden, kHidden);
        lw.ffn_norm = static_cast<const float*>(ptr(k++));
        lw.w_gate_up = take_linear(2 * kInter, kHidden);
        lw.w_down = take_linear(kHidden, kInter);
    }
    CUDALLM_CHECK(k == table.size(), "internal: tensor pointer assignment mismatch");

    config_.wtype = wt;
    config_.hidden = kHidden;
    config_.inter = kInter;
    config_.n_layers = kLayers;
    config_.n_heads = kHeads;
    config_.n_kv_heads = kKvHeads;
    config_.head_dim = kHeadDim;
    config_.vocab = kVocab;
    config_.max_seq_len = static_cast<int>(h.max_seq_len);
    config_.group_size = static_cast<int>(h.group_size);
    config_.rms_eps = h.rms_eps;
}

Weights::~Weights() {
    if (base_) {
        cudaError_t err = cudaFree(base_);
        if (err != cudaSuccess) std::fprintf(stderr, "cudallm: cudaFree failed in ~Weights: %s\n", cudaGetErrorString(err));
    }
}

const ModelConfig& Weights::config() const { return config_; }

uint64_t Weights::lm_and_layer_bytes() const { return total_bytes_; }

}  // namespace cudallm
