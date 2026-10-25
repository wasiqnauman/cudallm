// cudallm_bench: end-to-end decode benchmark (DESIGN.md §13.3).
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "common/cuda_check.h"
#include "common/cuda_timer.h"
#include "engine/engine.h"
#include "tools/args.h"
#include "tools/json_writer.h"

using namespace cudallm;

namespace {

const char* kUsage =
    "usage: cudallm_bench --model PATH [--prompt-len 128] [--gen 256] [--reps 5] [--warmup 1]\n"
    "                     [--matvec opt|naive] [--attn opt|naive] [--fuse 0|1] [--graph 0|1] [--rpb 0]\n"
    "                     [--json PATH]";

int prompt_token(int i) { return i == 0 ? 128000 : (i * 7919) % 120000 + 1000; }

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

}  // namespace

int main(int argc, char** argv) {
    Args args(argc, argv, kUsage,
              {"model", "prompt-len", "gen", "reps", "warmup", "matvec", "attn", "fuse", "graph", "rpb", "json"});
    if (!args.has("model")) args.fail("--model is required");
    const std::string model = args.get_str("model", "");
    const int prompt_len = args.get_int("prompt-len", 128);
    const int gen = args.get_int("gen", 256);
    const int reps = args.get_int("reps", 5);
    const int warmup = args.get_int("warmup", 1);
    const std::string matvec = args.get_str("matvec", "opt");
    const std::string attn = args.get_str("attn", "opt");
    if (prompt_len < 1) args.fail("--prompt-len must be >= 1");
    if (gen < 1 || reps < 1 || warmup < 0) args.fail("--gen and --reps must be >= 1, --warmup >= 0");
    if ((matvec != "opt" && matvec != "naive") || (attn != "opt" && attn != "naive")) {
        args.fail("--matvec/--attn must be opt or naive");
    }

    try {
        EngineOptions opts;
        opts.matvec = matvec == "opt" ? MatvecImpl::OPT : MatvecImpl::NAIVE;
        opts.attn = attn == "opt" ? AttnImpl::OPT : AttnImpl::NAIVE;
        opts.fuse = args.get_bool("fuse", true);
        opts.cuda_graph = args.get_bool("graph", true);
        opts.matvec_rpb = args.get_int("rpb", 0);
        Engine eng(model, opts);
        const ModelConfig& c = eng.config();
        const int max_seq = opts.max_seq_len ? opts.max_seq_len : c.max_seq_len;
        CUDALLM_CHECK(prompt_len + gen <= max_seq, "prompt-len + gen exceeds max_seq_len");
        const int rpb = opts.matvec_rpb ? opts.matvec_rpb : default_rpb(c.wtype);

        // Timing uses events on the legacy stream around a fully drained engine stream, so the
        // interval covers exactly the gen decode steps plus one host synchronize.
        CudaTimer timer;
        std::vector<double> tok_s, ms_per_tok;
        for (int r = 0; r < warmup + reps; ++r) {
            eng.reset();
            for (int i = 0; i < prompt_len; ++i) eng.step(prompt_token(i), nullptr, nullptr);
            eng.synchronize();
            timer.start(0);
            for (int i = 0; i < gen; ++i) eng.step(-1, nullptr, nullptr);
            eng.synchronize();
            const float ms = timer.stop_ms(0);
            if (r >= warmup) {
                tok_s.push_back(gen / (ms / 1e3));
                ms_per_tok.push_back(ms / gen);
            }
        }

        const uint64_t weight_bytes = eng.weight_bytes();
        const uint64_t kv_bytes = static_cast<uint64_t>(c.n_layers) * 2 * c.n_kv_heads * c.head_dim * 2 *
                                  (static_cast<uint64_t>(prompt_len) + (gen + 1) / 2);
        const uint64_t bytes_per_token = weight_bytes + kv_bytes;
        const double tok_s_med = median(tok_s);
        const double gbps = static_cast<double>(bytes_per_token) * tok_s_med / 1e9;
        cudaDeviceProp prop{};
        int dev = 0;
        CUDALLM_CUDA_CHECK(cudaGetDevice(&dev));
        CUDALLM_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        std::printf("%s %s matvec=%s attn=%s fuse=%d graph=%d rpb=%d prompt=%d gen=%d: %.2f tok/s (median of %d), "
                    "%.3f ms/token, %.1f GB/s\n",
                    model.c_str(), to_string(c.wtype), matvec.c_str(), attn.c_str(), opts.fuse, opts.cuda_graph, rpb,
                    prompt_len, gen, tok_s_med, reps, median(ms_per_tok), gbps);

        if (args.has("json")) {
            JsonWriter j;
            j.begin_object();
            j.key("tool").value("cudallm_bench");
            j.key("model").value(model);
            j.key("weight_dtype").value(to_string(c.wtype));
            j.key("options").begin_object();
            j.key("matvec").value(matvec);
            j.key("attn").value(attn);
            j.key("fuse").value(opts.fuse);
            j.key("graph").value(opts.cuda_graph);
            j.key("rpb").value(rpb);
            j.end_object();
            j.key("prompt_len").value(prompt_len);
            j.key("gen").value(gen);
            j.key("reps").value(reps);
            j.key("decode_tok_s").begin_array();
            for (double v : tok_s) j.value(v);
            j.end_array();
            j.key("decode_tok_s_median").value(tok_s_med);
            j.key("ms_per_token_median").value(median(ms_per_tok));
            j.key("weight_bytes").value(weight_bytes);
            j.key("kv_bytes_per_token_avg").value(kv_bytes);
            j.key("bytes_per_token").value(bytes_per_token);
            j.key("achieved_gbps_median").value(gbps);
            j.key("gpu_name").value(prop.name);
            j.end_object();
            j.write_file(args.get_str("json", ""));
        }
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "cudallm_bench: %s\n", ex.what());
        return 1;
    }
    return 0;
}
