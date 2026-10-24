"""Generate golden HF outputs for the 32-token golden prompt (DESIGN.md §12.6).

Usage: python -m cudallm.golden --hf-dir DIR --out tests/.cache/golden_hf.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GOLDEN_TEXT = ("Rivers carve valleys slowly, and engineers measure everything twice before trusting a number. "
               "The lighthouse keeper wrote one careful line in the logbook every night for forty years.")
N_GOLDEN = 32


def golden_ids(hf_dir: Path) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    ids = tokenizer(GOLDEN_TEXT).input_ids
    assert len(ids) >= N_GOLDEN, f"golden text has only {len(ids)} tokens"
    return ids[:N_GOLDEN]


def generate(hf_dir: Path, out: Path) -> None:
    ids = golden_ids(hf_dir)
    model = AutoModelForCausalLM.from_pretrained(hf_dir, torch_dtype=torch.float32, attn_implementation="eager")
    model.eval()
    with torch.no_grad():
        res = model(torch.tensor([ids], dtype=torch.int64), output_hidden_states=True)
    # hidden_states[16] has the final norm applied, so only 0..15 are residual-stream values.
    hidden = torch.stack([res.hidden_states[k][0] for k in range(16)]).numpy().astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, input_ids=np.array(ids, dtype=np.int64), logits=res.logits[0].numpy().astype(np.float32),
             hidden=hidden)
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate golden HF outputs")
    ap.add_argument("--hf-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    generate(args.hf_dir, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
