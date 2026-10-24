#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! .venv/bin/huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir models/hf/Llama-3.2-1B-Instruct --include "*.json" "*.safetensors"; then
    echo "Accept the license and run huggingface-cli login"
    exit 1
fi
