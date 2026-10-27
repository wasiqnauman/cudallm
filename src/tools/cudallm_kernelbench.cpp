// cudallm_kernelbench: kernel microbenchmarks.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "common/cuda_check.h"
#include "common/cuda_timer.h"
#include "kernels/kernels.h"
#include "tools/args.h"
#include "tools/json_writer.h"

using namespace cudallm;

namespace {

const char* kUsage =
    "usage: cudallm_kernelbench --op matvec --wtype bf16|int8|int4 --rows R --cols C --rpb N\n"
    "                           [--epilogue store|add|swiglu] [--fuse-norm 0|1] [--impl opt|naive] [--iters 200]\n"
    "       cudallm_kernelbench --op shapes --wtype W --rpb N [--impl opt|naive] [--iters 200]\n"
    "       cudallm_kernelbench --op attention --impl naive|opt --ctx N [--iters 200]\n"
    "       (all ops) [--json PATH]";

constexpr int kWarmupLaunches = 20;
constexpr int kAttnMaxSeq = 4096;
constexpr int kNh = 32, kNkv = 8, kHd = 64;

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

// 20 warmup launches, then `iters` launches each timed with events; returns the median in µs.
template <typename F>
double time_median_us(F&& launch, int iters) {
    for (int i = 0; i < kWarmupLaunches; ++i) launch();
    CUDALLM_CUDA_CHECK(cudaDeviceSynchronize());
    CudaTimer timer;
    std::vector<double> us;
    us.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        timer.start(0);
        launch();
        us.push_back(timer.stop_ms(0) * 1e3);
    }
    return median(us);
}

class DevMem {
public:
    explicit DevMem(size_t bytes) : bytes_(bytes) {
        if (bytes) CUDALLM_CUDA_CHECK(cudaMalloc(&p_, bytes));
    }
    ~DevMem() {
        if (p_) cudaFree(p_);
    }
    DevMem(const DevMem&) = delete;
    DevMem& operator=(const DevMem&) = delete;
    template <typename T>
    T* as() const { return static_cast<T*>(p_); }
    void upload(const void* src) { CUDALLM_CUDA_CHECK(cudaMemcpy(p_, src, bytes_, cudaMemcpyHostToDevice)); }

private:
    void* p_ = nullptr;
    size_t bytes_;
};

uint16_t f32_to_bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

WeightType parse_wtype(const Args& args, const std::string& s) {
    if (s == "bf16") return WeightType::BF16;
    if (s == "int8") return WeightType::INT8;
    if (s == "int4") return WeightType::INT4;
    args.fail("--wtype must be bf16, int8 or int4");
}

Epilogue parse_epilogue(const Args& args, const std::string& s) {
    if (s == "store") return Epilogue::STORE;
    if (s == "add") return Epilogue::ADD;
    if (s == "swiglu") return Epilogue::SWIGLU;
    args.fail("--epilogue must be store, add or swiglu");
}

const char* epilogue_name(Epilogue e) {
    switch (e) {
        case Epilogue::STORE: return "store";
        case Epilogue::ADD: return "add";
        case Epilogue::SWIGLU: return "swiglu";
    }
    return "?";
}

struct MatvecResult {
    std::string name;
    int rows, cols;
    std::string impl;
    WeightType wtype;
    Epilogue ep;
    bool fuse_norm;
    int rpb;
    double median_us, gbps;
    uint64_t bytes;
};

// Builds random operands (seed 1234) for one matvec shape and times it.
MatvecResult bench_matvec(const std::string& name, bool opt, WeightType wt, Epilogue ep, bool fuse_norm, int rpb,
                          int rows, int cols, int iters) {
    std::mt19937 rng(1234);
    std::normal_distribution<float> wdist(0.f, 0.02f), xdist(0.f, 1.f);
    std::uniform_real_distribution<float> ndist(0.5f, 1.5f), sdist(0.5f / 127.f, 1.5f / 127.f);
    std::uniform_int_distribution<int> i8dist(-127, 127), u8dist(0, 255);

    const size_t n = static_cast<size_t>(rows) * cols;
    size_t w_bytes = 0, s_bytes = 0;
    std::vector<uint8_t> w_host, s_host;
    if (wt == WeightType::BF16) {
        w_bytes = n * 2;
        w_host.resize(w_bytes);
        auto* p = reinterpret_cast<uint16_t*>(w_host.data());
        for (size_t i = 0; i < n; ++i) p[i] = f32_to_bf16_bits(wdist(rng));
    } else if (wt == WeightType::INT8) {
        w_bytes = n;
        w_host.resize(w_bytes);
        for (size_t i = 0; i < n; ++i) w_host[i] = static_cast<uint8_t>(static_cast<int8_t>(i8dist(rng)));
        s_bytes = sizeof(float) * rows;
        s_host.resize(s_bytes);
        auto* s = reinterpret_cast<float*>(s_host.data());
        for (int r = 0; r < rows; ++r) s[r] = sdist(rng);
    } else {
        w_bytes = n / 2;
        w_host.resize(w_bytes);
        for (size_t i = 0; i < w_bytes; ++i) w_host[i] = static_cast<uint8_t>(u8dist(rng));
        const size_t n_scales = static_cast<size_t>(rows) * (cols / 128);
        s_bytes = sizeof(__half) * n_scales;
        s_host.resize(s_bytes);
        auto* s = reinterpret_cast<__half*>(s_host.data());
        for (size_t i = 0; i < n_scales; ++i) s[i] = __float2half(sdist(rng) * 127.f / 7.f);
    }
    std::vector<float> x(cols), norm_w(cols);
    for (auto& v : x) v = xdist(rng);
    for (auto& v : norm_w) v = ndist(rng);
    const int out_rows = ep == Epilogue::SWIGLU ? rows / 2 : rows;

    DevMem dw(w_bytes), ds(s_bytes), dx(sizeof(float) * cols), dn(sizeof(float) * cols), dy(sizeof(float) * out_rows);
    dw.upload(w_host.data());
    if (s_bytes) ds.upload(s_host.data());
    dx.upload(x.data());
    dn.upload(norm_w.data());
    CUDALLM_CUDA_CHECK(cudaMemset(dy.as<void>(), 0, sizeof(float) * out_rows));

    MatvecArgs a{};
    a.w = dw.as<void>();
    a.scales = s_bytes ? ds.as<void>() : nullptr;
    a.x = dx.as<float>();
    a.norm_w = fuse_norm ? dn.as<float>() : nullptr;
    a.eps = 1e-5f;
    a.y = dy.as<float>();
    a.rows = rows;
    a.cols = cols;
    const double us = time_median_us(
        [&] {
            if (opt) {
                launch_matvec_opt(wt, ep, fuse_norm, a, rpb, 0);
            } else {
                launch_matvec_naive(a, 0);
            }
        },
        iters);
    const uint64_t bytes = w_bytes + s_bytes;
    return {name, rows, cols, opt ? "opt" : "naive", wt, ep, fuse_norm, rpb, us, bytes / (us / 1e6) / 1e9, bytes};
}

void print_matvec(const MatvecResult& r) {
    std::printf("%-8s %6d x %-5d %-5s %-4s %-6s fuse_norm=%d rpb=%d: %10.2f us  %7.1f GB/s\n", r.name.c_str(), r.rows,
                r.cols, r.impl.c_str(), to_string(r.wtype), epilogue_name(r.ep), r.fuse_norm, r.rpb, r.median_us, r.gbps);
}

void json_matvec(JsonWriter& j, const MatvecResult& r) {
    j.begin_object();
    j.key("name").value(r.name);
    j.key("rows").value(r.rows);
    j.key("cols").value(r.cols);
    j.key("impl").value(r.impl);
    j.key("wtype").value(to_string(r.wtype));
    j.key("epilogue").value(epilogue_name(r.ep));
    j.key("fuse_norm").value(r.fuse_norm);
    j.key("rpb").value(r.rpb);
    j.key("bytes").value(r.bytes);
    j.key("median_us").value(r.median_us);
    j.key("gbps").value(r.gbps);
    j.end_object();
}

double bench_attention(bool opt, int ctx, int iters) {
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.f, 1.f);
    const size_t cache_n = static_cast<size_t>(kNkv) * kAttnMaxSeq * kHd;
    std::vector<float> q(kNh * kHd);
    std::vector<__half> k(cache_n), v(cache_n);
    for (auto& e : q) e = dist(rng);
    for (auto& e : k) e = __float2half(dist(rng));
    for (auto& e : v) e = __float2half(dist(rng));
    const int n_split = kAttnMaxSeq / kAttnChunk;
    DevMem dq(sizeof(float) * q.size()), dk(sizeof(__half) * cache_n), dv(sizeof(__half) * cache_n), dpos(sizeof(int)),
        dout(sizeof(float) * kNh * kHd), dpo(sizeof(float) * kNh * n_split * kHd), dpml(sizeof(float) * kNh * n_split * 2);
    dq.upload(q.data());
    dk.upload(k.data());
    dv.upload(v.data());
    const int pos = ctx - 1;
    dpos.upload(&pos);
    return time_median_us(
        [&] {
            if (opt) {
                launch_attention_opt(dq.as<float>(), dk.as<__half>(), dv.as<__half>(), dpos.as<int>(), dpo.as<float>(),
                                     dpml.as<float>(), dout.as<float>(), kNh, kNkv, kHd, kAttnMaxSeq, 0);
            } else {
                launch_attention_naive(dq.as<float>(), dk.as<__half>(), dv.as<__half>(), dpos.as<int>(), dout.as<float>(),
                                       kNh, kNkv, kHd, kAttnMaxSeq, 0);
            }
        },
        iters);
}

}  // namespace

int main(int argc, char** argv) {
    Args args(argc, argv, kUsage, {"op", "wtype", "rows", "cols", "rpb", "epilogue", "fuse-norm", "impl", "iters", "ctx", "json"});
    const std::string op = args.get_str("op", "");
    const int iters = args.get_int("iters", 200);
    if (iters < 1) args.fail("--iters must be >= 1");
    try {
        cudaDeviceProp prop{};
        int dev = 0;
        CUDALLM_CUDA_CHECK(cudaGetDevice(&dev));
        CUDALLM_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        JsonWriter j;
        j.begin_object();
        j.key("tool").value("cudallm_kernelbench");
        j.key("op").value(op);
        j.key("gpu_name").value(prop.name);
        j.key("iters").value(iters);

        if (op == "matvec" || op == "shapes") {
            const std::string impl = args.get_str("impl", "opt");
            if (impl != "opt" && impl != "naive") args.fail("--impl must be opt or naive");
            const bool opt = impl == "opt";
            const WeightType wt = parse_wtype(args, args.get_str("wtype", "bf16"));
            const int rpb = args.get_int("rpb", default_rpb(wt));
            j.key("impl").value(impl);
            j.key("wtype").value(to_string(wt));
            j.key("rpb").value(rpb);
            j.key("results").begin_array();
            if (op == "matvec") {
                if (!args.has("rows") || !args.has("cols")) args.fail("--rows and --cols are required for --op matvec");
                const Epilogue ep = parse_epilogue(args, args.get_str("epilogue", "store"));
                const bool fuse_norm = args.get_bool("fuse-norm", false);
                const MatvecResult r = bench_matvec("matvec", opt, wt, ep, fuse_norm, rpb, args.get_int("rows", 0),
                                                    args.get_int("cols", 0), iters);
                print_matvec(r);
                json_matvec(j, r);
                j.end_array();
            } else {
                // The five model shapes with the fused engine's epilogues. The naive kernel only
                // supports STORE without a fused norm, so naive runs use that for every shape.
                struct Shape {
                    const char* name;
                    int rows, cols;
                    Epilogue ep;
                    bool fuse_norm;
                };
                const Shape shapes[] = {{"qkv", 3072, 2048, Epilogue::STORE, true},
                                        {"wo", 2048, 2048, Epilogue::ADD, false},
                                        {"gate_up", 16384, 2048, Epilogue::SWIGLU, true},
                                        {"down", 2048, 8192, Epilogue::ADD, false},
                                        {"lm_head", 128256, 2048, Epilogue::STORE, true}};
                double us[5];
                for (int s = 0; s < 5; ++s) {
                    const Shape& sh = shapes[s];
                    const MatvecResult r = bench_matvec(sh.name, opt, wt, opt ? sh.ep : Epilogue::STORE,
                                                        opt && sh.fuse_norm, rpb, sh.rows, sh.cols, iters);
                    print_matvec(r);
                    json_matvec(j, r);
                    us[s] = r.median_us;
                }
                j.end_array();
                const double weighted = 16.0 * (us[0] + us[1] + us[2] + us[3]) + us[4];
                std::printf("weighted_us = 16*(qkv+wo+gate_up+down) + lm_head = %.2f\n", weighted);
                j.key("weighted_us").value(weighted);
            }
        } else if (op == "attention") {
            const std::string impl = args.get_str("impl", "opt");
            if (impl != "opt" && impl != "naive") args.fail("--impl must be opt or naive");
            const int ctx = args.get_int("ctx", 0);
            if (ctx < 1 || ctx > kAttnMaxSeq) args.fail("--ctx must be in [1, 4096]");
            const double us = bench_attention(impl == "opt", ctx, iters);
            std::printf("attention %-5s ctx=%d: %.2f us\n", impl.c_str(), ctx, us);
            j.key("impl").value(impl);
            j.key("ctx").value(ctx);
            j.key("median_us").value(us);
        } else {
            args.fail("--op must be matvec, shapes or attention");
        }
        j.end_object();
        if (args.has("json")) j.write_file(args.get_str("json", ""));
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "cudallm_kernelbench: %s\n", ex.what());
        return 1;
    }
    return 0;
}
