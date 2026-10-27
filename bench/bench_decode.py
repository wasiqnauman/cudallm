"""End-to-end decode benchmark suites around build/cudallm_bench.

Usage: python bench/bench_decode.py --suite {ablation,quant,context,tg,all} [--names A0_naive,...]
"""
import argparse
import json
import tempfile
from pathlib import Path

from common import BUILD_DIR, MODEL_PATHS, RESULTS_DIR, git_commit, gpu_snapshot, load_peak_gbps, run, save_json

# (name, dtype, matvec, attn, fuse, graph, prompt_len, gen, reps)
ABLATION = [
    ("A0_naive", "bf16", "naive", "naive", 0, 0, 128, 256, 5),
    ("A1_matvec_opt", "bf16", "opt", "naive", 0, 0, 128, 256, 5),
    ("A2_attn_opt", "bf16", "opt", "opt", 0, 0, 128, 256, 5),
    ("A3_fused", "bf16", "opt", "opt", 1, 0, 128, 256, 5),
    ("A4_graph_nofuse", "bf16", "opt", "opt", 0, 1, 128, 256, 5),
    ("A5_full", "bf16", "opt", "opt", 1, 1, 128, 256, 5),
]
QUANT = [(f"Q_{d}", d, "opt", "opt", 1, 1, 128, 256, 5) for d in ("bf16", "int8", "int4")]
CONTEXT = [run_ for c in (128, 512, 1024, 2048, 4000)
           for run_ in ((f"ctx_opt_{c}", "bf16", "opt", "opt", 1, 1, c, 32, 3),
                        (f"ctx_naiveattn_{c}", "bf16", "opt", "naive", 1, 1, c, 32, 3))]
TG = [(f"tg_{d}", d, "opt", "opt", 1, 1, 1, 256, 5) for d in ("bf16", "int8", "int4")]
SUITES = {"ablation": ABLATION, "quant": QUANT, "context": CONTEXT, "tg": TG}


def run_one(suite: str, entry: tuple, peak: float) -> None:
    name, dtype, matvec, attn, fuse, graph, prompt_len, gen, reps = entry
    with tempfile.TemporaryDirectory() as tmp:
        tmp_json = Path(tmp) / "bench.json"
        gpu_before = gpu_snapshot()
        run([BUILD_DIR / "cudallm_bench", "--model", MODEL_PATHS[dtype], "--prompt-len", prompt_len, "--gen", gen,
             "--reps", reps, "--matvec", matvec, "--attn", attn, "--fuse", fuse, "--graph", graph, "--json", tmp_json])
        gpu_after = gpu_snapshot()
        with open(tmp_json) as f:
            data = json.load(f)
    data["name"] = name
    data["git_commit"] = git_commit()
    data["gpu_before"] = gpu_before
    data["gpu_after"] = gpu_after
    data["pct_of_peak"] = 100.0 * data["achieved_gbps_median"] / peak
    save_json(RESULTS_DIR / "decode" / suite / f"{name}.json", data)


def main() -> None:
    ap = argparse.ArgumentParser(description="cudallm decode benchmark suites")
    ap.add_argument("--suite", choices=(*SUITES, "all"), required=True)
    ap.add_argument("--names", default="", help="comma-separated subset of run names (default: the whole suite)")
    args = ap.parse_args()
    wanted = {n for n in args.names.split(",") if n}
    peak = load_peak_gbps()
    suites = list(SUITES) if args.suite == "all" else [args.suite]
    known = {e[0] for s in suites for e in SUITES[s]}
    unknown = wanted - known
    if unknown:
        raise SystemExit(f"unknown run names for suite {args.suite}: {sorted(unknown)}")
    for suite in suites:
        for entry in SUITES[suite]:
            if not wanted or entry[0] in wanted:
                run_one(suite, entry, peak)


if __name__ == "__main__":
    main()
