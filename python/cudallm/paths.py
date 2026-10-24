"""Filesystem locations and token ids shared by the package, tests and benchmarks."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HF_DIR = REPO_ROOT / "models/hf/Llama-3.2-1B-Instruct"
MODEL_PATHS = {"bf16": REPO_ROOT / "models/llama-3.2-1b-instruct.bf16.cudallm",
               "int8": REPO_ROOT / "models/llama-3.2-1b-instruct.int8.cudallm",
               "int4": REPO_ROOT / "models/llama-3.2-1b-instruct.int4.cudallm"}
LIB_PATH = Path(os.environ.get("CUDALLM_LIB", REPO_ROOT / "build/libcudallm.so"))
BUILD_DIR = REPO_ROOT / "build"
RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = REPO_ROOT / "tests/.cache"
BOS_ID = 128000
EOS_IDS = (128001, 128008, 128009)
