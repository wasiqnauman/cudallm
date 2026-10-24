import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch

from cudallm.paths import CACHE_DIR, HF_DIR, LIB_PATH, MODEL_PATHS, REPO_ROOT

GOLDEN_PATH = CACHE_DIR / "golden_hf.npz"


@pytest.fixture(scope="session")
def model_path() -> Callable[[str], Path]:
    def get(dtype: str) -> Path:
        p = MODEL_PATHS[dtype]
        if not p.exists():
            pytest.skip(f"model file missing: {p} (run scripts/export_all.sh)")
        return p
    return get


@pytest.fixture(scope="session")
def golden() -> dict[str, np.ndarray]:
    if not GOLDEN_PATH.exists():
        if not HF_DIR.exists():
            pytest.skip(f"HF checkpoint missing: {HF_DIR} (run scripts/download_model.sh)")
        # A subprocess so the fp32 HF model's memory is released before the tests run.
        env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "python"))
        subprocess.run([sys.executable, "-m", "cudallm.golden", "--hf-dir", str(HF_DIR), "--out", str(GOLDEN_PATH)],
                       check=True, env=env, cwd=REPO_ROOT)
    with np.load(GOLDEN_PATH) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="session")
def ref_outputs(model_path, golden) -> Callable[[str], tuple[np.ndarray, np.ndarray]]:
    """dtype -> (logits [32, V], residuals [32, n_layers+1, H]) from ReferenceModel with an fp16 KV cache."""
    from cudallm.reference import ReferenceModel

    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def get(dtype: str) -> tuple[np.ndarray, np.ndarray]:
        if dtype not in cache:
            ref = ReferenceModel(model_path(dtype), kv_dtype=torch.float16)
            logits, residuals = [], []
            for t in golden["input_ids"].tolist():
                lg, res = ref.step(int(t))
                logits.append(lg.numpy())
                residuals.append(torch.stack(res).numpy())
            cache[dtype] = (np.stack(logits), np.stack(residuals))
        return cache[dtype]
    return get


@pytest.fixture(scope="session")
def lib_available() -> Path:
    if not LIB_PATH.exists():
        pytest.skip(f"library missing: {LIB_PATH} (run scripts/build.sh)")
    return LIB_PATH
