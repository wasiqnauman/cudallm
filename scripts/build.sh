#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# gcc-13 / CUDA 12.6 paths: approved deviation for native Ubuntu 26.10 (PROGRESS.md "Decisions").
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++-13 \
    -DCMAKE_CUDA_HOST_COMPILER=g++-13 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
    && cmake --build build -j"$(nproc)"
