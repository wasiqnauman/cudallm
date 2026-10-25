"""Engine vs ReferenceModel on the 32 golden tokens (DESIGN.md §14.7).

After Phase 1 the only allowed change to this file is appending to ENABLED_CONFIGS.
"""
import numpy as np
import pytest

from cudallm.engine import Engine

pytestmark = pytest.mark.model

# (dtype, matvec, attn, fuse, graph)
ENABLED_CONFIGS = [
    ("bf16", "naive", "naive", 0, 0),
    ("bf16", "opt", "naive", 0, 0),
]

N_DETERMINISM = 4


def _cfg_id(cfg) -> str:
    return "-".join(str(c) for c in cfg)


def _run(eng: Engine, ids: list[int], graph: int):
    logits, residuals, nexts = [], [], []
    for t in ids:
        if graph:
            nxt, lg = eng.step(t, want_logits=True)
            res = None
        else:
            nxt, lg, res = eng.step_debug(t)
        logits.append(lg)
        residuals.append(res)
        nexts.append(nxt)
    return np.stack(logits), (None if graph else np.stack(residuals)), nexts


@pytest.fixture(scope="module")
def engine_runs(model_path, golden):
    """cfg -> dict(logits [32,V], residuals [32,17,H] or None, nexts, rerun_logits [4,V])."""
    cache: dict[tuple, dict] = {}

    def get(cfg) -> dict:
        if cfg not in cache:
            dtype, matvec, attn, fuse, graph = cfg
            ids = [int(t) for t in golden["input_ids"]]
            with Engine(model_path(dtype), matvec=matvec, attn=attn, fuse=bool(fuse), cuda_graph=bool(graph)) as eng:
                logits, residuals, nexts = _run(eng, ids, graph)
                eng.reset()
                rerun_logits, _, _ = _run(eng, ids[:N_DETERMINISM], graph)
            cache[cfg] = {"logits": logits, "residuals": residuals, "nexts": nexts, "rerun_logits": rerun_logits}
        return cache[cfg]
    return get


@pytest.mark.parametrize("cfg", [c for c in ENABLED_CONFIGS if not c[4]], ids=_cfg_id)
def test_residuals(cfg, engine_runs, ref_outputs):
    run = engine_runs(cfg)
    _, ref_res = ref_outputs(cfg[0])
    eng_res = run["residuals"]
    assert eng_res.shape == ref_res.shape
    for p in range(ref_res.shape[0]):
        for k in range(ref_res.shape[1]):
            a, b = eng_res[p, k].astype(np.float64), ref_res[p, k].astype(np.float64)
            rel = np.linalg.norm(a - b) / np.linalg.norm(b)
            assert rel < 1e-3, f"position {p}, residual index {k}: relative error {rel:.3e}"


@pytest.mark.parametrize("cfg", ENABLED_CONFIGS, ids=_cfg_id)
def test_logits(cfg, engine_runs, ref_outputs):
    run = engine_runs(cfg)
    ref_logits, _ = ref_outputs(cfg[0])
    diff = np.max(np.abs(run["logits"] - ref_logits))
    assert diff < 2e-2, f"max |engine - reference| logit difference {diff:.3e}"
    agree = int(np.sum(np.argmax(run["logits"], axis=1) == np.argmax(ref_logits, axis=1)))
    assert agree >= 31, f"argmax agrees at only {agree}/32 positions"


@pytest.mark.parametrize("cfg", ENABLED_CONFIGS, ids=_cfg_id)
def test_next_token_is_argmax(cfg, engine_runs):
    run = engine_runs(cfg)
    assert run["nexts"] == [int(np.argmax(lg)) for lg in run["logits"]]


@pytest.mark.parametrize("cfg", ENABLED_CONFIGS, ids=_cfg_id)
def test_reset_determinism(cfg, engine_runs):
    run = engine_runs(cfg)
    np.testing.assert_array_equal(run["rerun_logits"], run["logits"][:N_DETERMINISM])
