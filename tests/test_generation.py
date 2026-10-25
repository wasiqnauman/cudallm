"""Greedy generation: engine vs ReferenceModel token-for-token (DESIGN.md §14.8).

After Phase 1 the only allowed change to this file is appending to ENABLED_CONFIGS.
"""
import pytest
import torch
from transformers import AutoTokenizer

from cudallm.engine import Engine
from cudallm.paths import HF_DIR
from cudallm.reference import ReferenceModel

pytestmark = pytest.mark.model

# (dtype, matvec, attn, fuse, graph)
ENABLED_CONFIGS = [
    ("bf16", "naive", "naive", 0, 0),
]

PROMPT = "What is the capital of France? Answer in one word."
N_NEW = 32


def _cfg_id(cfg) -> str:
    return "-".join(str(c) for c in cfg)


@pytest.fixture(scope="module")
def tokenizer():
    if not HF_DIR.exists():
        pytest.skip(f"HF checkpoint missing: {HF_DIR} (run scripts/download_model.sh)")
    return AutoTokenizer.from_pretrained(HF_DIR)


@pytest.fixture(scope="module")
def prompt_ids(tokenizer) -> list[int]:
    return list(tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True))


@pytest.fixture(scope="module")
def ref_greedy(model_path, prompt_ids):
    cache: dict[str, list[int]] = {}

    def get(dtype: str) -> list[int]:
        if dtype not in cache:
            cache[dtype] = ReferenceModel(model_path(dtype), kv_dtype=torch.float16).greedy(prompt_ids, N_NEW)
        return cache[dtype]
    return get


def _engine_greedy(eng: Engine, prompt_ids: list[int], n_new: int) -> list[int]:
    nxt = None
    for t in prompt_ids:
        nxt, _ = eng.step(t)
    out = [nxt]
    while len(out) < n_new:
        nxt, _ = eng.step(None)
        out.append(nxt)
    return out


@pytest.mark.parametrize("cfg", ENABLED_CONFIGS, ids=_cfg_id)
def test_greedy_matches_reference(cfg, model_path, prompt_ids, ref_greedy, tokenizer):
    dtype, matvec, attn, fuse, graph = cfg
    with Engine(model_path(dtype), matvec=matvec, attn=attn, fuse=bool(fuse), cuda_graph=bool(graph)) as eng:
        got = _engine_greedy(eng, prompt_ids, N_NEW)
    assert got == ref_greedy(dtype)
    if dtype == "bf16":
        text = tokenizer.decode(got)
        assert "Paris" in text, f"decoded text does not contain 'Paris': {text!r}"
