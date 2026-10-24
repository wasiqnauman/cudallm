import numpy as np
import pytest
from safetensors import safe_open
from transformers import AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from cudallm import format as fmt
from cudallm.paths import HF_DIR, MODEL_PATHS

pytestmark = pytest.mark.model

EXPECTED_HEADER = {"magic": b"CLLM", "version": 1, "hidden_size": 2048, "intermediate_size": 8192, "n_layers": 16,
                   "n_heads": 32, "n_kv_heads": 8, "head_dim": 64, "vocab_size": 128256, "max_seq_len": 4096,
                   "reserved0": 0, "table_offset": 256}
WEIGHT_DTYPE = {"bf16": fmt.WEIGHT_BF16, "int8": fmt.WEIGHT_INT8, "int4": fmt.WEIGHT_INT4}


def _existing_models() -> list[tuple[str, object]]:
    return [(d, p) for d, p in MODEL_PATHS.items() if p.exists()]


def _hf_tensor(key: str) -> np.ndarray:
    for f in sorted(HF_DIR.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as sf:
            if key in sf.keys():
                return sf.get_tensor(key)
    raise KeyError(key)


def test_header_and_table():
    models = _existing_models()
    if not models:
        pytest.skip("no exported model files (run scripts/export_all.sh)")
    for dtype, path in models:
        hdr = fmt.read_header(path)
        for k, v in EXPECTED_HEADER.items():
            assert hdr[k] == v, f"{path}: {k}"
        assert hdr["weight_dtype"] == WEIGHT_DTYPE[dtype]
        assert hdr["rms_eps"] == np.float32(1e-5)
        assert hdr["rope_theta"] == np.float32(500000.0)
        assert hdr["group_size"] == (128 if dtype == "int4" else 0)
        quant = dtype != "bf16"
        assert hdr["n_tensors"] == (164 if quant else 99)

        table = fmt.read_table(path)
        assert [e["name"] for e in table] == fmt.expected_names(16, quant)
        prev_end = 256 + 80 * len(table)
        for e in table:
            assert e["offset"] % 256 == 0
            assert e["offset"] >= prev_end
            assert e["nbytes"] == fmt.expected_nbytes(e["dtype"], e["rows"], e["cols"])
            prev_end = e["offset"] + e["nbytes"]
        assert path.stat().st_size == table[-1]["offset"] + table[-1]["nbytes"]


@pytest.fixture(scope="module")
def bf16_table(model_path):
    path = model_path("bf16")
    return path, {e["name"]: e for e in fmt.read_table(path)}


def _bits(t) -> np.ndarray:
    import torch
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


def test_bf16_wqkv_layer0(bf16_table):
    path, table = bf16_table
    got = fmt.load_tensor(path, table["layers.0.wqkv.weight"])
    want = np.concatenate([_bits(_hf_tensor(f"model.layers.0.self_attn.{n}_proj.weight")) for n in "qkv"], axis=0)
    assert got.shape == (3072, 2048)
    np.testing.assert_array_equal(got, want)


def test_bf16_gate_up_layer3(bf16_table):
    path, table = bf16_table
    got = fmt.load_tensor(path, table["layers.3.w_gate_up.weight"])
    want = np.concatenate([_bits(_hf_tensor(f"model.layers.3.mlp.{n}_proj.weight")) for n in ("gate", "up")], axis=0)
    assert got.shape == (16384, 2048)
    np.testing.assert_array_equal(got, want)


def test_bf16_inv_freq(bf16_table):
    path, table = bf16_table
    config = AutoConfig.from_pretrained(HF_DIR)
    inv_freq, _ = ROPE_INIT_FUNCTIONS["llama3"](config, "cpu")
    got = fmt.load_tensor(path, table["rope.inv_freq"])
    assert np.array_equal(got, inv_freq.float().numpy())


def test_bf16_final_norm(bf16_table):
    path, table = bf16_table
    got = fmt.load_tensor(path, table["norm.weight"])
    want = _hf_tensor("model.norm.weight").float().numpy()
    assert np.array_equal(got, want)
