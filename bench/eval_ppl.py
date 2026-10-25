"""WikiText-2 perplexity for the engine or HF transformers (DESIGN.md §15.7).

Usage: python bench/eval_ppl.py --target {engine,hf} [--dtype bf16|int8|int4] [--windows 40] [--window-len 512]
"""
import argparse
import math

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODEL_PATHS, RESULTS_DIR, git_commit, gpu_snapshot, save_json

from cudallm.engine import Engine
from cudallm.paths import BOS_ID, HF_DIR

HF_PPL_RANGE = (6.0, 25.0)


def nll_from_logits(logits: np.ndarray, target: int) -> float:
    """-log softmax(logits)[target] in float64."""
    z = logits.astype(np.float64)
    m = z.max()
    return float(m + math.log(np.exp(z - m).sum()) - z[target])


def load_windows(tokenizer, n_windows: int, window_len: int) -> list[list[int]]:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text).input_ids
    if ids and ids[0] == BOS_ID:
        ids = ids[1:]
    assert len(ids) >= n_windows * window_len, f"only {len(ids)} tokens for {n_windows} windows"
    return [ids[k * window_len:(k + 1) * window_len] for k in range(n_windows)]


def eval_engine(dtype: str, windows: list[list[int]]) -> list[float]:
    """Per-window mean NLL. step(BOS) predicts w[0]; step(w[j]) predicts w[j+1]."""
    out = []
    with Engine(MODEL_PATHS[dtype]) as eng:
        for w in windows:
            eng.reset()
            _, logits = eng.step(BOS_ID, want_logits=True)
            nll = [nll_from_logits(logits, w[0])]
            for j in range(len(w) - 1):
                _, logits = eng.step(w[j], want_logits=True)
                nll.append(nll_from_logits(logits, w[j + 1]))
            out.append(float(np.mean(nll)))
    return out


def eval_hf(windows: list[list[int]]) -> list[float]:
    model = AutoModelForCausalLM.from_pretrained(HF_DIR, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to("cuda").eval()
    out = []
    with torch.no_grad():
        for w in windows:
            inp = torch.tensor([[BOS_ID] + w], dtype=torch.int64, device="cuda")
            logits = model(inp).logits[0, :-1].to(torch.float64)
            nll = torch.nn.functional.cross_entropy(logits, torch.tensor(w, device="cuda"), reduction="none")
            out.append(float(nll.mean().item()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="WikiText-2 perplexity")
    ap.add_argument("--target", choices=("engine", "hf"), required=True)
    ap.add_argument("--dtype", choices=("bf16", "int8", "int4"), default="bf16")
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--window-len", type=int, default=512)
    args = ap.parse_args()
    if args.target == "hf" and args.dtype != "bf16":
        raise SystemExit("HF perplexity is measured in bf16 only")

    tokenizer = AutoTokenizer.from_pretrained(HF_DIR)
    windows = load_windows(tokenizer, args.windows, args.window_len)
    gpu_before = gpu_snapshot()
    window_nll = eval_engine(args.dtype, windows) if args.target == "engine" else eval_hf(windows)
    # Equal-length windows, so the mean of window means is the mean over all tokens.
    ppl = math.exp(float(np.mean(window_nll)))
    if args.target == "hf" and not HF_PPL_RANGE[0] <= ppl <= HF_PPL_RANGE[1]:
        raise RuntimeError(f"HF bf16 PPL {ppl:.3f} outside sanity range {HF_PPL_RANGE}: tokenization/protocol bug?")
    print(f"{args.target} {args.dtype}: PPL {ppl:.4f} over {len(windows) * args.window_len} tokens")
    save_json(RESULTS_DIR / "ppl" / f"{args.target}_{args.dtype}.json", {
        "tool": "eval_ppl", "target": args.target, "dtype": args.dtype, "ppl": ppl,
        "n_tokens": len(windows) * args.window_len, "windows": len(windows), "window_len": args.window_len,
        "window_mean_nll": window_nll, "dataset": "Salesforce/wikitext wikitext-2-raw-v1 test",
        "git_commit": git_commit(), "gpu_before": gpu_before, "gpu_after": gpu_snapshot(),
    })


if __name__ == "__main__":
    main()
