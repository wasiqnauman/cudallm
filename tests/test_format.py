import struct

import numpy as np
import pytest

from cudallm import format as fmt

pytestmark = pytest.mark.kernels


def _tiny_tensors(rng: np.random.Generator) -> list[tuple[str, int, int, int, bytes]]:
    f32 = rng.standard_normal((5, 1)).astype(np.float32)
    bf16 = rng.integers(0, 2**16, size=(3, 7), dtype=np.uint16)
    f16 = rng.standard_normal((2, 3)).astype(np.float16)
    i8 = rng.integers(-127, 128, size=(4, 9), dtype=np.int8)
    i4 = rng.integers(0, 256, size=(3, 4), dtype=np.uint8)  # logical cols = 8
    return [
        ("a.f32", fmt.F32, 5, 1, f32.tobytes()),
        ("b.bf16", fmt.BF16, 3, 7, bf16.tobytes()),
        ("c.f16", fmt.F16, 2, 3, f16.tobytes()),
        ("d.i8", fmt.I8, 4, 9, i8.tobytes()),
        ("e.i4", fmt.I4X2, 3, 8, i4.tobytes()),
    ]


HEADER = {"weight_dtype": fmt.WEIGHT_BF16, "hidden_size": 2048, "intermediate_size": 8192, "n_layers": 16,
          "n_heads": 32, "n_kv_heads": 8, "head_dim": 64, "vocab_size": 128256, "max_seq_len": 4096,
          "rms_eps": 1e-5, "rope_theta": 500000.0, "group_size": 0}


def test_struct_sizes():
    assert struct.calcsize(fmt.HEADER_FMT) == 72
    assert fmt.HEADER_SIZE == 256
    assert struct.calcsize(fmt.ENTRY_FMT) == 80
    assert fmt.ENTRY_SIZE == 80


def test_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    tensors = _tiny_tensors(rng)
    path = tmp_path / "tiny.cudallm"
    fmt.write_cudallm(path, HEADER, tensors)

    hdr = fmt.read_header(path)
    assert hdr["magic"] == b"CLLM"
    assert hdr["version"] == 1
    assert hdr["n_tensors"] == len(tensors)
    assert hdr["table_offset"] == 256
    assert hdr["reserved0"] == 0
    for k, v in HEADER.items():
        if isinstance(v, float):
            assert hdr[k] == pytest.approx(v, rel=1e-7)
        else:
            assert hdr[k] == v
    raw = path.read_bytes()
    assert raw[72:256] == b"\0" * 184

    table = fmt.read_table(path)
    assert [e["name"] for e in table] == [t[0] for t in tensors]
    prev_end = 256 + 80 * len(tensors)
    for e, (name, dtype, rows, cols, data) in zip(table, tensors):
        assert e["offset"] % 256 == 0
        assert e["offset"] >= prev_end
        assert raw[prev_end:e["offset"]] == b"\0" * (e["offset"] - prev_end)
        assert (e["dtype"], e["rows"], e["cols"], e["reserved"]) == (dtype, rows, cols, 0)
        assert e["nbytes"] == fmt.expected_nbytes(dtype, rows, cols) == len(data)
        assert raw[e["offset"]:e["offset"] + e["nbytes"]] == data
        prev_end = e["offset"] + e["nbytes"]
    assert len(raw) == table[-1]["offset"] + table[-1]["nbytes"]

    by_name = {e["name"]: e for e in table}
    a = fmt.load_tensor(path, by_name["a.f32"])
    assert a.dtype == np.float32 and a.shape == (5,)
    np.testing.assert_array_equal(a, np.frombuffer(tensors[0][4], dtype=np.float32))
    b = fmt.load_tensor(path, by_name["b.bf16"])
    assert b.dtype == np.uint16 and b.shape == (3, 7)
    c = fmt.load_tensor(path, by_name["c.f16"])
    assert c.dtype == np.float16 and c.shape == (2, 3)
    d = fmt.load_tensor(path, by_name["d.i8"])
    assert d.dtype == np.int8 and d.shape == (4, 9)
    e = fmt.load_tensor(path, by_name["e.i4"])
    assert e.dtype == np.uint8 and e.shape == (3, 4)
    np.testing.assert_array_equal(e.tobytes(), tensors[4][4])


def test_write_rejects_wrong_nbytes(tmp_path):
    with pytest.raises(AssertionError):
        fmt.write_cudallm(tmp_path / "bad.cudallm", HEADER, [("x", fmt.F32, 4, 1, b"\0" * 12)])


def test_bf16_bits_to_f32():
    vals = np.array([1.0, -2.5, 0.0, 3.140625], dtype=np.float32)
    bits = (vals.view(np.uint32) >> 16).astype(np.uint16)
    np.testing.assert_array_equal(fmt.bf16_bits_to_f32(bits), vals)


def test_expected_names():
    bf16 = fmt.expected_names(16, False)
    quant = fmt.expected_names(16, True)
    assert len(bf16) == 99
    assert len(quant) == 164
    assert bf16[:4] == ["rope.inv_freq", "tok_embeddings.weight", "norm.weight", "layers.0.attn_norm.weight"]
    assert quant[:5] == ["rope.inv_freq", "tok_embeddings.weight", "tok_embeddings.scales", "norm.weight",
                         "layers.0.attn_norm.weight"]
    assert bf16[-1] == "layers.15.w_down.weight"
    assert quant[-1] == "layers.15.w_down.scales"
