"""Single-kernel tests against float64 numpy references."""
import numpy as np
import pytest

from cudallm import engine as E
from cudallm.quantize import (dequantize_int4_groups, dequantize_int8_rows, quantize_int4_groups,
                              quantize_int8_rows)

pytestmark = pytest.mark.kernels

NH, NKV, HD, MAX_SEQ = 32, 8, 64, 4096


@pytest.fixture(autouse=True)
def _needs_lib(lib_available):
    return lib_available


def f32_to_bf16_bits(a: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even fp32 -> bf16 bits."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_bits_to_f64(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def llama3_inv_freq() -> np.ndarray:
    """RoPE inverse-frequency formula, in float64 then stored as fp32 (tests only need plausible positive values)."""
    i = np.arange(32, dtype=np.float64)
    base = 1.0 / 500000.0 ** (2 * i / 64)
    lam = 2 * np.pi / base
    out = np.where(lam < 8192 / 4, base, np.where(lam > 8192 / 1, base / 32, 0.0))
    mid = (lam >= 8192 / 4) & (lam <= 8192 / 1)
    s = (8192 / lam - 1) / (4 - 1)
    out = np.where(mid, (1 - s) * base / 32 + s * base, out)
    return out.astype(np.float32)


# ---- matvec --------------------------------------------------------------------------------

def test_matvec_naive_bf16_store():
    rows, cols = 1024, 2048
    rng = np.random.default_rng(10)
    bits = f32_to_bf16_bits(rng.normal(0, 0.02, size=(rows, cols)))
    x = rng.standard_normal(cols).astype(np.float32)
    y = E.kernel_matvec("naive", "bf16", "store", 0, 0, bits, None, x, None, 1e-5, None, rows, cols)
    ref = bf16_bits_to_f64(bits) @ x.astype(np.float64)
    assert np.allclose(y, ref, rtol=1e-4, atol=1e-4)


def _matvec_operands(wtype: str, rows: int, cols: int, seed: int):
    """Returns (stored weight, scales or None, exact dequantized fp32 weight as float64)."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.02, size=(rows, cols)).astype(np.float32)
    if wtype == "bf16":
        bits = f32_to_bf16_bits(w)
        return bits, None, bf16_bits_to_f64(bits)
    if wtype == "int8":
        q, scales = quantize_int8_rows(w)
        return q, scales, dequantize_int8_rows(q, scales).astype(np.float64)
    packed, scales = quantize_int4_groups(w, 128)
    return packed, scales, dequantize_int4_groups(packed, scales).astype(np.float64)


@pytest.mark.parametrize("shape", [(1024, 2048), (2048, 8192)], ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("rpb", [1, 2, 4, 8])
@pytest.mark.parametrize("fuse_norm", [0, 1])
@pytest.mark.parametrize("epilogue", ["store", "add", "swiglu"])
@pytest.mark.parametrize("wtype", ["bf16", "int8", "int4"])
def test_matvec_opt(wtype, epilogue, fuse_norm, rpb, shape):
    rows, cols = shape
    eps = 1e-5
    w, scales, w64 = _matvec_operands(wtype, rows, cols, seed=20 + rows + cols)
    rng = np.random.default_rng(21 + rows)
    x = rng.standard_normal(cols).astype(np.float32)
    norm_w = rng.uniform(0.5, 1.5, cols).astype(np.float32)
    out_rows = rows // 2 if epilogue == "swiglu" else rows
    y_init = rng.standard_normal(out_rows).astype(np.float32) if epilogue == "add" else None

    y = E.kernel_matvec("opt", wtype, epilogue, fuse_norm, rpb, w, scales, x, norm_w if fuse_norm else None, eps,
                        y_init, rows, cols)

    x64 = x.astype(np.float64)
    if fuse_norm:
        x64 = x64 / np.sqrt(np.mean(x64 * x64) + eps) * norm_w.astype(np.float64)
    d = w64 @ x64
    if epilogue == "swiglu":
        g, u = d[:out_rows], d[out_rows:]
        ref = g / (1 + np.exp(-g)) * u
    elif epilogue == "add":
        ref = y_init.astype(np.float64) + d
    else:
        ref = d
    assert np.allclose(y, ref, rtol=1e-4, atol=1e-4)


# ---- rmsnorm -------------------------------------------------------------------------------

def test_rmsnorm():
    n, eps = 2048, 1e-5
    rng = np.random.default_rng(11)
    x = rng.standard_normal(n).astype(np.float32)
    w = rng.uniform(0.5, 1.5, n).astype(np.float32)
    y = E.kernel_rmsnorm(x, w, eps)
    x64 = x.astype(np.float64)
    ref = x64 / np.sqrt(np.mean(x64 * x64) + eps) * w.astype(np.float64)
    assert np.allclose(y, ref, rtol=1e-5, atol=1e-5)


# ---- rope + kv write -----------------------------------------------------------------------

SENTINEL = np.float16(-7.25)


def _rope_ref(v: np.ndarray, inv_freq: np.ndarray, pos: int) -> np.ndarray:
    """v: [n, HD] float64. Angle computed in fp32 exactly like the kernel; rotation in float64."""
    ang = (np.float32(pos) * inv_freq).astype(np.float64)
    c, s = np.cos(ang), np.sin(ang)
    a, b = v[:, :HD // 2], v[:, HD // 2:]
    return np.concatenate([a * c - b * s, b * c + a * s], axis=1)


def _ulp_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a - b| in fp16 ulps, for same-sign values (signed bit patterns mapped to a monotone line)."""
    def ordered(x):
        u = x.astype(np.float16).view(np.uint16).astype(np.int32)
        return np.where(u & 0x8000, 0x8000 - u, u)
    return np.abs(ordered(a) - ordered(b))


@pytest.mark.parametrize("pos", [0, 1, 777, 4095])
@pytest.mark.parametrize("fused", [0, 1])
def test_rope_kv(fused, pos):
    rng = np.random.default_rng(12 + pos)
    inv_freq = llama3_inv_freq()
    qkv = rng.standard_normal((NH + 2 * NKV) * HD).astype(np.float32)
    k_cache = np.full((NKV, MAX_SEQ, HD), SENTINEL, dtype=np.float16)
    v_cache = np.full((NKV, MAX_SEQ, HD), SENTINEL, dtype=np.float16)

    out_qkv, k_out, v_out = E.kernel_rope_kv(fused, qkv, inv_freq, pos, k_cache, v_cache, NH, NKV, HD, MAX_SEQ)
    k_out = k_out.reshape(NKV, MAX_SEQ, HD)
    v_out = v_out.reshape(NKV, MAX_SEQ, HD)

    q64 = qkv[:NH * HD].astype(np.float64).reshape(NH, HD)
    k64 = qkv[NH * HD:(NH + NKV) * HD].astype(np.float64).reshape(NKV, HD)
    v32 = qkv[(NH + NKV) * HD:].reshape(NKV, HD)
    q_ref = _rope_ref(q64, inv_freq, pos)
    k_ref = _rope_ref(k64, inv_freq, pos)

    assert np.allclose(out_qkv[:NH * HD].reshape(NH, HD), q_ref, rtol=0, atol=1e-5)
    if not fused:
        assert np.allclose(out_qkv[NH * HD:(NH + NKV) * HD].reshape(NKV, HD), k_ref, rtol=0, atol=1e-5)
        np.testing.assert_array_equal(out_qkv[(NH + NKV) * HD:], qkv[(NH + NKV) * HD:])

    assert np.all(_ulp_diff(k_out[:, pos, :], k_ref.astype(np.float16)) <= 1)
    np.testing.assert_array_equal(v_out[:, pos, :], v32.astype(np.float16))

    others = np.ones(MAX_SEQ, dtype=bool)
    others[pos] = False
    assert np.all(k_out[:, others, :] == SENTINEL)
    assert np.all(v_out[:, others, :] == SENTINEL)


# ---- attention -----------------------------------------------------------------------------

def _attention_ref(q: np.ndarray, k: np.ndarray, v: np.ndarray, ctx: int) -> np.ndarray:
    q64 = q.astype(np.float64).reshape(NH, HD)
    k64 = k[:, :ctx].astype(np.float64)
    v64 = v[:, :ctx].astype(np.float64)
    out = np.empty((NH, HD))
    for h in range(NH):
        kvh = h // (NH // NKV)
        s = (k64[kvh] @ q64[h]) / 8.0
        p = np.exp(s - s.max())
        p /= p.sum()
        out[h] = p @ v64[kvh]
    return out.reshape(-1)


@pytest.fixture(scope="module")
def attn_data():
    rng = np.random.default_rng(13)
    q = rng.standard_normal(NH * HD).astype(np.float32)
    k = rng.standard_normal((NKV, MAX_SEQ, HD)).astype(np.float16)
    v = rng.standard_normal((NKV, MAX_SEQ, HD)).astype(np.float16)
    return q, k, v


@pytest.mark.parametrize("ctx", [1, 7, 128, 129, 1000, 4096])
@pytest.mark.parametrize("impl", ["naive", "opt"])
def test_attention(impl, ctx, attn_data):
    q, k, v = attn_data
    out = E.kernel_attention(impl, q, k, v, ctx, NH, NKV, HD, MAX_SEQ)
    assert np.allclose(out, _attention_ref(q, k, v, ctx), rtol=1e-4, atol=1e-4)


# ---- silu_mul ------------------------------------------------------------------------------

def test_silu_mul():
    inter = 8192
    rng = np.random.default_rng(14)
    gu = (rng.standard_normal(2 * inter) * 2).astype(np.float32)
    act = E.kernel_silu_mul(gu)
    g, u = gu[:inter].astype(np.float64), gu[inter:].astype(np.float64)
    assert np.allclose(act, g / (1 + np.exp(-g)) * u, rtol=1e-5)


# ---- argmax --------------------------------------------------------------------------------

VOCAB = 128256


def test_argmax_random():
    logits = np.random.default_rng(15).standard_normal(VOCAB).astype(np.float32)
    assert E.kernel_argmax(logits) == int(np.argmax(logits))


def test_argmax_tie_lowest_index():
    logits = np.random.default_rng(16).standard_normal(VOCAB).astype(np.float32)
    logits[5] = logits[100000] = 50.0
    assert E.kernel_argmax(logits) == 5


def test_argmax_last_index():
    logits = np.random.default_rng(17).standard_normal(VOCAB).astype(np.float32)
    logits[-1] = 50.0
    assert E.kernel_argmax(logits) == VOCAB - 1
