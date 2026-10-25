# cudallm

cudallm is a from-scratch C++17/CUDA inference engine for **meta-llama/Llama-3.2-1B-Instruct** at batch size 1. It loads the model from a custom single-file weight format, runs the whole decode step as hand-written CUDA kernels (no cuBLAS, cuDNN, CUTLASS or PyTorch in the engine), captures the step as a CUDA graph, and supports bf16, int8 and int4 weights. A Python package wraps it through a small C API (ctypes) and provides the exporter, a PyTorch reference model, tests and benchmarks.

The project's goal was single-stream decode close to the GPU's memory-bandwidth limit, with every optimization measured and recorded. On an RTX 3060 the final bf16 engine reaches about 90% of the measured peak DRAM bandwidth; the full story, including what did not work, is in [docs/WRITEUP.md](docs/WRITEUP.md) and [docs/OPTIMIZATION_LOG.md](docs/OPTIMIZATION_LOG.md).

## Hardware

NVIDIA GeForce RTX 3060 (12 GB, sm_86, 28 SMs, rated 360 GB/s), driver 580, CUDA 12.6 toolkit with gcc-13 as host compiler. The original design targeted WSL2 Ubuntu 24.04; this build runs on native Ubuntu 26.10 with the approved setup changes recorded in `PROGRESS.md`.

## Quickstart

```bash
bash scripts/setup_env.sh                       # needs sudo: gcc-13, CUDA 12.6 toolkit, Python 3.12 venv
.venv/bin/huggingface-cli login                 # after accepting the Llama 3.2 license on Hugging Face
bash scripts/check_env.sh
bash scripts/download_model.sh
bash scripts/build.sh
bash scripts/export_all.sh                      # bf16, int8, int4 .cudallm files in models/
bash scripts/run_tests.sh -p no:cacheprovider   # full test suite
PYTHONPATH=python .venv/bin/python -m cudallm.generate --prompt "What is the capital of France? Answer in one word."
.venv/bin/python bench/bench_bandwidth.py && .venv/bin/python bench/bench_decode.py --suite all
.venv/bin/python bench/make_report.py           # regenerates docs/RESULTS.md and docs/img/
```

## Results

Copied from the top table of [docs/RESULTS.md](docs/RESULTS.md) (generated from `results/`; re-copy whenever the report is regenerated):

| Metric | Value | Source |
|---|---|---|
| cudallm bf16 decode (A5, prompt 128, gen 256) | 124.80 tok/s | decode/ablation/A5_full.json |
| cudallm bf16 % of measured peak DRAM bandwidth | 90.48 % | decode/ablation/A5_full.json |
| cudallm int4 decode (prompt 128, gen 256) | 383.24 tok/s | decode/quant/Q_int4.json |
| HF transformers bf16 (sdpa, eager, prompt 128) | 90.02 tok/s | compare/hf_bf16.json |
| llama.cpp best of f16/bf16 (tg256, f16) | 123.07 tok/s | compare/llamacpp_f16.json |
| Speedup cudallm A5 vs HF (prompt 128) | 1.39x | — |
| Speedup cudallm bf16 vs llama.cpp best f16/bf16 (tg256) | 1.01x | decode/tg/tg_bf16.json |
| Speedup cudallm int4 vs cudallm bf16 (A5 options) | 3.10x | decode/quant/ |

## Repository map

- `include/` — public C API header (`cudallm/cudallm.h`).
- `src/` — the engine: weight loader, engine, CUDA kernels, C API, and the benchmark tools (`cudallm_bench`, `cudallm_kernelbench`, `cudallm_bandwidth`).
- `python/` — the `cudallm` package: exporter, quantizer, PyTorch reference, golden-output generator, ctypes engine wrapper, `generate`.
- `tests/` — pytest suite: format, quantization, export, reference vs HF, kernels, engine vs reference, generation.
- `bench/` — benchmark, sweep, perplexity, comparison and report scripts.
- `scripts/` — environment setup/check, model download, build, export, test and profiling helpers.
- `results/` — JSON outputs of the benchmark scripts (the only source of numbers in `docs/`).
- `docs/` — design document, optimization log, generated results report, writeup and plots.

## Further reading

- [docs/WRITEUP.md](docs/WRITEUP.md) — technical writeup (draft).
- [docs/OPTIMIZATION_LOG.md](docs/OPTIMIZATION_LOG.md) — every optimization attempt with before/after measurements and profiler evidence.
- [docs/RESULTS.md](docs/RESULTS.md) — generated results report.
