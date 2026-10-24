#pragma once

#include <cuda_runtime.h>

#include "common/cuda_check.h"

namespace cudallm {

class CudaTimer {
public:
    CudaTimer() {
        CUDALLM_CUDA_CHECK(cudaEventCreate(&start_));
        CUDALLM_CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    void start(cudaStream_t stream) { CUDALLM_CUDA_CHECK(cudaEventRecord(start_, stream)); }

    float stop_ms(cudaStream_t stream) {
        CUDALLM_CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDALLM_CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDALLM_CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

}  // namespace cudallm
