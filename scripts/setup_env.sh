#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Native Ubuntu 26.10 variant of DESIGN.md §6.2 (approved deviation, see PROGRESS.md "Decisions").
# The NVIDIA driver is already installed; only the CUDA 12.6 toolkit is installed here.
sudo apt-get update
sudo apt-get install -y build-essential gcc-13 g++-13 cmake git wget
# CUDA 12.6.3 toolkit only (no driver); builds use gcc-13 as the host compiler (scripts/build.sh).
# Ubuntu 26.10 has no libxml2.so.2, so NVIDIA's cuda-installer cannot run: unpack the runfile
# (makeself) and copy the toolkit components it would install. Work on disk, /tmp is a small tmpfs.
CUDA_RUN=cuda_12.6.3_560.35.05_linux.run
CUDA_TMP="$HOME/.cache/cudallm-cuda-install"
CUDA_DIR=/usr/local/cuda-12.6
mkdir -p "$CUDA_TMP"
[ -f "$CUDA_TMP/$CUDA_RUN" ] || wget -q "https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/$CUDA_RUN" -O "$CUDA_TMP/$CUDA_RUN"
# The marker is written last, so a run that failed part-way redoes the whole step.
if [ ! -f "$CUDA_DIR/.cudallm-toolkit-ok" ]; then
    rm -rf "$CUDA_TMP/x"
    mkdir -p "$CUDA_TMP/x"
    (cd "$CUDA_TMP/x" && sh "../$CUDA_RUN" --tar xf)
    sudo mkdir -p "$CUDA_DIR"
    for c in cuda_cccl cuda_cudart cuda_cuobjdump cuda_cupti cuda_cuxxfilt cuda_nvcc cuda_nvdisasm cuda_nvml_dev \
             cuda_nvprune cuda_nvrtc cuda_nvtx cuda_profiler_api cuda_sanitizer_api; do
        sudo cp -a "$CUDA_TMP/x/builds/$c/." "$CUDA_DIR/"
    done
    sudo mkdir -p "$CUDA_DIR/nsight-compute" "$CUDA_DIR/nsight-systems"
    sudo cp -a "$CUDA_TMP/x/builds/nsight_compute/." "$CUDA_DIR/nsight-compute/"
    sudo cp -a "$CUDA_TMP/x/builds/nsight_systems/." "$CUDA_DIR/nsight-systems/"
    sudo ln -sf ../nsight-compute/ncu "$CUDA_DIR/bin/ncu"
    sudo ln -sf ../nsight-systems/bin/nsys "$CUDA_DIR/bin/nsys"
    # glibc >= 2.41 declares these noexcept; CUDA 12.6 does not, and nvcc rejects the mismatch.
    sudo sed -i -E 's/^(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ (double|float) +(rsqrtf?|sinpif?|cospif?)\((double|float) x\));$/\1 noexcept (true);/' \
        "$CUDA_DIR/targets/x86_64-linux/include/crt/math_functions.h"
    [ "$(grep -c 'x) noexcept (true);' "$CUDA_DIR/targets/x86_64-linux/include/crt/math_functions.h")" -eq 6 ] \
        || { echo "math_functions.h patch did not apply to all 6 declarations" >&2; exit 1; }
    sudo touch "$CUDA_DIR/.cudallm-toolkit-ok"
fi
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
