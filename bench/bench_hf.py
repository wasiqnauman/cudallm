"""HF transformers bf16 decode baseline on the same GPU (DESIGN.md §15.5)."""
import statistics
import time

import torch
import transformers
from transformers import AutoModelForCausalLM

from common import RESULTS_DIR, git_commit, gpu_snapshot, save_json

from cudallm.paths import HF_DIR

WARMUP = 1
REPS = 5
GEN = 256


def synthetic_ids(prompt_len: int) -> list[int]:
    """Same forced prompt as cudallm_bench."""
    return [128000 if i == 0 else (i * 7919) % 120000 + 1000 for i in range(prompt_len)]


def timed_generate(model, input_ids: torch.Tensor, n_new: int) -> float:
    attn = torch.ones_like(input_ids)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(input_ids, attention_mask=attn, max_new_tokens=n_new, min_new_tokens=n_new, do_sample=False,
                   pad_token_id=model.config.eos_token_id[0])
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench(model, prompt_len: int, name: str) -> None:
    input_ids = torch.tensor([synthetic_ids(prompt_len)], dtype=torch.int64, device="cuda")
    gpu_before = gpu_snapshot()
    tok_s, t1s, t257s = [], [], []
    for rep in range(WARMUP + REPS):
        t1 = timed_generate(model, input_ids, 1)
        t257 = timed_generate(model, input_ids, GEN + 1)
        if rep >= WARMUP:
            t1s.append(t1)
            t257s.append(t257)
            tok_s.append(GEN / (t257 - t1))
    save_json(RESULTS_DIR / "compare" / f"{name}.json", {
        "tool": "bench_hf", "name": name, "model": str(HF_DIR), "dtype": "bf16", "attn_implementation": "sdpa",
        "prompt_len": prompt_len, "gen": GEN, "reps": REPS,
        "decode_tok_s": tok_s, "decode_tok_s_median": statistics.median(tok_s),
        "t1_s": t1s, "t257_s": t257s,
        "transformers": transformers.__version__, "torch": torch.__version__,
        "git_commit": git_commit(), "gpu_before": gpu_before, "gpu_after": gpu_snapshot(),
        "gpu_name": torch.cuda.get_device_name(0),
    })


def main() -> None:
    model = AutoModelForCausalLM.from_pretrained(HF_DIR, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to("cuda").eval()
    with torch.no_grad():
        bench(model, 128, "hf_bf16")
        bench(model, 1, "hf_bf16_tg")


if __name__ == "__main__":
    main()
