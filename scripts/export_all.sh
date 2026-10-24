#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
force=0
[ "${1:-}" = "--force" ] && force=1
for dtype in bf16 int8 int4; do
    out="models/llama-3.2-1b-instruct.${dtype}.cudallm"
    if [ -f "$out" ] && [ "$force" -eq 0 ]; then
        echo "skip $dtype: $out exists (use --force to overwrite)"
        continue
    fi
    PYTHONPATH=python .venv/bin/python -m cudallm.export --hf-dir models/hf/Llama-3.2-1B-Instruct --out "$out" --dtype "$dtype"
done
