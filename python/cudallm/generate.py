"""Greedy text generation with the cudallm engine.

Usage: python -m cudallm.generate --prompt TEXT [--dtype bf16|int8|int4] [--max-new-tokens 128] [--no-chat]
       [--matvec opt|naive] [--attn opt|naive] [--no-fuse] [--no-graph]
"""
import argparse
import sys
import time

from transformers import AutoTokenizer

from .engine import Engine
from .paths import EOS_IDS, HF_DIR, MODEL_PATHS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Greedy generation with cudallm")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--dtype", choices=("bf16", "int8", "int4"), default="bf16")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--no-chat", action="store_true")
    ap.add_argument("--matvec", choices=("opt", "naive"), default="opt")
    ap.add_argument("--attn", choices=("opt", "naive"), default="opt")
    ap.add_argument("--no-fuse", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(HF_DIR)
    if args.no_chat:
        ids = list(tokenizer(args.prompt).input_ids)
    else:
        ids = list(tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], add_generation_prompt=True))

    with Engine(MODEL_PATHS[args.dtype], matvec=args.matvec, attn=args.attn, fuse=not args.no_fuse,
                cuda_graph=not args.no_graph) as eng:
        nxt = None
        for t in ids:
            nxt, _ = eng.step(t)
        generated: list[int] = []
        printed = ""
        t0 = time.perf_counter()
        while True:
            if nxt in EOS_IDS:
                break
            generated.append(nxt)
            # Decode the whole sequence each time so multi-token characters print correctly.
            text = tokenizer.decode(generated)
            sys.stdout.write(text[len(printed):])
            sys.stdout.flush()
            printed = text
            if len(generated) >= args.max_new_tokens:
                break
            nxt, _ = eng.step(None)
        elapsed = time.perf_counter() - t0
    tok_s = len(generated) / elapsed if elapsed > 0 else 0.0
    print(f"\n[prompt_tokens={len(ids)} generated={len(generated)} decode_tok_s={tok_s:.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
