import numpy as np
import pytest
import torch

from cudallm.reference import ReferenceModel

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def ref_fp32_kv(model_path, golden):
    ref = ReferenceModel(model_path("bf16"), kv_dtype=torch.float32)
    logits, residuals = [], []
    for t in golden["input_ids"].tolist():
        lg, res = ref.step(int(t))
        logits.append(lg.numpy())
        residuals.append(torch.stack(res).numpy())
    return np.stack(logits), np.stack(residuals)  # [32, V], [32, 17, H]


def test_residuals_match_hf(ref_fp32_kv, golden):
    _, ref_res = ref_fp32_kv
    hf_hidden = golden["hidden"]  # [16, 32, H]
    for p in range(32):
        for k in range(16):
            a, b = ref_res[p][k].astype(np.float64), hf_hidden[k][p].astype(np.float64)
            rel = np.linalg.norm(a - b) / np.linalg.norm(b)
            assert rel < 1e-4, f"position {p} layer index {k}: relative error {rel:.3e}"


def test_logits_match_hf(ref_fp32_kv, golden):
    ref_logits, _ = ref_fp32_kv
    hf_logits = golden["logits"]
    assert np.max(np.abs(ref_logits - hf_logits)) < 1e-3
    np.testing.assert_array_equal(np.argmax(ref_logits, axis=1), np.argmax(hf_logits, axis=1))
