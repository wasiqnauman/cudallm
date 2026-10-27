#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace cudallm {

// Typed errors so the C API can map failures to distinct error codes.
class FormatError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
class IoError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
class ArgError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

}  // namespace cudallm

#define CUDALLM_CUDA_CHECK(expr)                                                         \
    do {                                                                                 \
        cudaError_t cudallm_err_ = (expr);                                               \
        if (cudallm_err_ != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" +    \
                                     std::to_string(__LINE__) + ": " + #expr + ": " +    \
                                     cudaGetErrorString(cudallm_err_));                  \
        }                                                                                \
    } while (0)

#define CUDALLM_CHECK(cond, msg)                 \
    do {                                         \
        if (!(cond)) {                           \
            throw std::runtime_error(msg);       \
        }                                        \
    } while (0)
