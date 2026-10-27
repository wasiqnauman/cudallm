#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# llama.cpp comparison: latest master, built for sm_86, tg256 for f16/bf16/q8_0.
mkdir -p results/compare
LLAMA_DIR=${LLAMA_DIR:-$HOME/llama.cpp}
[ -d "$LLAMA_DIR" ] || git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
# llama.cpp links cuBLAS, which the project's CUDA install omits (cudallm itself must not use it).
# Fetch NVIDIA's cuBLAS redist matching CUDA 12.6.3 into the llama.cpp tree only (no sudo).
CUBLAS_VER=12.6.4.1
CUBLAS_DIR="$LLAMA_DIR/libcublas-linux-x86_64-$CUBLAS_VER-archive"
if [ ! -d "$CUBLAS_DIR" ]; then
  CUBLAS_TXZ="$LLAMA_DIR/libcublas-$CUBLAS_VER.tar.xz"
  wget -q "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/linux-x86_64/libcublas-linux-x86_64-$CUBLAS_VER-archive.tar.xz" -O "$CUBLAS_TXZ"
  echo "ec682bac6387f9cdfd0c20b25a16cd6ed0b8b3b7ff42be9eaeb41828e3a72572  $CUBLAS_TXZ" | sha256sum -c -
  tar -xJf "$CUBLAS_TXZ" -C "$LLAMA_DIR"
  rm -f "$CUBLAS_TXZ"
fi
# The link step and llama-bench need libcudart/libcublas on the loader path (setup_env.sh only adds
# the CUDA lib dir to ~/.bashrc, which non-bash shells do not read).
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:$CUBLAS_DIR/lib:${LD_LIBRARY_PATH:-}"
# gcc-13 / CUDA 12.6 paths, same toolchain as scripts/build.sh.
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_HOST_COMPILER=g++-13 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
    -DCUDA_cublas_LIBRARY="$CUBLAS_DIR/lib/libcublas.so" -DCUDA_cublasLt_LIBRARY="$CUBLAS_DIR/lib/libcublasLt.so" \
    -DCMAKE_CXX_FLAGS="-I$CUBLAS_DIR/include" -DCMAKE_CUDA_FLAGS="-I$CUBLAS_DIR/include"
cmake --build "$LLAMA_DIR/build" -j"$(nproc)" --target llama-bench
# Separate venv (Python 3.12 via uv, as for .venv: Ubuntu 26.10 lacks python3-venv) keeps llama.cpp's
# requirements away from our pinned torch.
[ -d "$LLAMA_DIR/.venv" ] || "$HOME/.local/bin/uv" venv --seed --python 3.12 "$LLAMA_DIR/.venv"
"$LLAMA_DIR/.venv/bin/pip" install -r "$LLAMA_DIR/requirements/requirements-convert_hf_to_gguf.txt"
for T in f16 bf16 q8_0; do
  OUT=models/llama-3.2-1b-instruct.$T.gguf
  [ -f "$OUT" ] || "$LLAMA_DIR/.venv/bin/python" "$LLAMA_DIR/convert_hf_to_gguf.py" models/hf/Llama-3.2-1B-Instruct --outtype $T --outfile "$OUT"
  "$LLAMA_DIR/build/bin/llama-bench" -m "$OUT" -p 0 -n 256 -ngl 99 -r 5 -o json > results/compare/llamacpp_$T.json || echo "llama-bench $T failed" >&2
done
git -C "$LLAMA_DIR" rev-parse --short HEAD > results/compare/llamacpp_commit.txt
