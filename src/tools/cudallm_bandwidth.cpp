// cudallm_bandwidth: measured DRAM bandwidth of this GPU.
// The project's "measured peak" is read_gbps_best.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <functional>
#include <vector>

#include "common/cuda_check.h"
#include "common/cuda_timer.h"
#include "kernels/kernels.h"
#include "tools/args.h"
#include "tools/json_writer.h"

using namespace cudallm;

namespace {

constexpr size_t kBytes = 1ull << 30;
constexpr int kWarmup = 3;
constexpr int kReps = 20;

struct Stat {
    double best, median;
};

// Times `fn` kReps times after kWarmup untimed runs; GB/s = bytes / s / 1e9.
Stat measure(const std::function<void()>& fn, double bytes) {
    CudaTimer timer;
    for (int i = 0; i < kWarmup; ++i) fn();
    CUDALLM_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> gbps;
    for (int i = 0; i < kReps; ++i) {
        timer.start(0);
        fn();
        const float ms = timer.stop_ms(0);
        gbps.push_back(bytes / (ms / 1e3) / 1e9);
    }
    std::sort(gbps.begin(), gbps.end());
    return {gbps.back(), (gbps[kReps / 2 - 1] + gbps[kReps / 2]) / 2.0};
}

}  // namespace

int main(int argc, char** argv) {
    Args args(argc, argv, "usage: cudallm_bandwidth [--json PATH]", {"json"});
    try {
        int dev = 0, n_sms = 0;
        CUDALLM_CUDA_CHECK(cudaGetDevice(&dev));
        CUDALLM_CUDA_CHECK(cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev));
        cudaDeviceProp prop{};
        CUDALLM_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        const int grid = n_sms * 32;
        const size_t n = kBytes / sizeof(uint4);

        uint4 *src = nullptr, *dst = nullptr;
        unsigned* sink = nullptr;
        CUDALLM_CUDA_CHECK(cudaMalloc(&src, kBytes));
        CUDALLM_CUDA_CHECK(cudaMalloc(&dst, kBytes));
        CUDALLM_CUDA_CHECK(cudaMalloc(&sink, sizeof(unsigned) * grid));
        CUDALLM_CUDA_CHECK(cudaMemset(src, 0x5a, kBytes));
        CUDALLM_CUDA_CHECK(cudaMemset(dst, 0, kBytes));

        const Stat rd = measure([&] { launch_bw_read(src, n, sink, grid, 0); }, kBytes);
        const Stat wr = measure([&] { launch_bw_write(dst, n, grid, 0); }, kBytes);
        const Stat cp = measure([&] { launch_bw_copy(src, dst, n, grid, 0); }, 2.0 * kBytes);
        const Stat mc = measure([&] { CUDALLM_CUDA_CHECK(cudaMemcpy(dst, src, kBytes, cudaMemcpyDeviceToDevice)); }, 2.0 * kBytes);

        std::printf("gpu: %s  grid=%d x 256\n", prop.name, grid);
        std::printf("read   best %.1f GB/s  median %.1f GB/s\n", rd.best, rd.median);
        std::printf("write  best %.1f GB/s  median %.1f GB/s\n", wr.best, wr.median);
        std::printf("copy   best %.1f GB/s  median %.1f GB/s (read+write)\n", cp.best, cp.median);
        std::printf("memcpy best %.1f GB/s (read+write)\n", mc.best);

        if (args.has("json")) {
            JsonWriter j;
            j.begin_object();
            j.key("tool").value("cudallm_bandwidth");
            j.key("read_gbps_best").value(rd.best);
            j.key("read_gbps_median").value(rd.median);
            j.key("write_gbps_best").value(wr.best);
            j.key("write_gbps_median").value(wr.median);
            j.key("copy_gbps_best").value(cp.best);
            j.key("copy_gbps_median").value(cp.median);
            j.key("memcpy_d2d_gbps_best").value(mc.best);
            j.key("gpu_name").value(prop.name);
            j.end_object();
            j.write_file(args.get_str("json", ""));
        }
        CUDALLM_CUDA_CHECK(cudaFree(src));
        CUDALLM_CUDA_CHECK(cudaFree(dst));
        CUDALLM_CUDA_CHECK(cudaFree(sink));
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "cudallm_bandwidth: %s\n", ex.what());
        return 1;
    }
    return 0;
}
