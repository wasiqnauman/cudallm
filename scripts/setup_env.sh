#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Native Ubuntu 26.10 variant of DESIGN.md §6.2 (approved deviation, see PROGRESS.md "Decisions").
# The NVIDIA driver is already installed; only the CUDA 12.6 toolkit is installed here.
sudo apt-get update
sudo apt-get install -y build-essential gcc-13 g++-13 cmake git wget
# CUDA 12.6.3 toolkit only (no driver). --override skips the default-GCC version check;
# builds use gcc-13 as the host compiler (scripts/build.sh). Extract on disk, /tmp is a small tmpfs.
CUDA_RUN=cuda_12.6.3_560.35.05_linux.run
CUDA_TMP="$HOME/.cache/cudallm-cuda-install"
mkdir -p "$CUDA_TMP"
[ -f "$CUDA_TMP/$CUDA_RUN" ] || wget -q "https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/$CUDA_RUN" -O "$CUDA_TMP/$CUDA_RUN"
[ -x /usr/local/cuda-12.6/bin/nvcc ] || sudo sh "$CUDA_TMP/$CUDA_RUN" --toolkit --silent --override --tmpdir="$CUDA_TMP"
grep -q 'cuda-12.6/bin' ~/.bashrc || echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
grep -q 'cuda-12.6/lib64' ~/.bashrc || echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}' >> ~/.bashrc
# Python 3.12 (Ubuntu 26.10 ships 3.14 only) via a pinned uv release.
[ -x "$HOME/.local/bin/uv" ] || wget -qO- https://astral.sh/uv/0.8.0/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh
"$HOME/.local/bin/uv" python install 3.12
"$HOME/.local/bin/uv" venv --seed --python 3.12 .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
rm -rf "$CUDA_TMP"
echo "Done. Run: source ~/.bashrc && source .venv/bin/activate && huggingface-cli login"
