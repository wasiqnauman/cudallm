import numpy as np
import pytest

from cudallm.quantize import (dequantize_int4_groups, dequantize_int8_rows, quantize_int4_groups,
                              quantize_int8_rows, unpack_int4)

pytestmark = pytest.mark.kernels


def test_int8_error_bound():
    rng = np.random.default_rng(1)
    w = rng.normal(0, 0.02, size=(64, 256)).astype(np.float32)
    q, scales = quantize_int8_rows(w)
    assert q.dtype == np.int8 and scales.dtype == np.float32 and scales.shape == (64,)
    assert q.min() >= -127 and q.max() <= 127
    err = np.abs(dequantize_int8_rows(q, scales) - w)
    assert np.all(err <= scales[:, None] / 2 + 1e-7)


def test_int8_zero_rows():
    rng = np.random.default_rng(2)
    w = rng.normal(0, 0.02, size=(4, 128)).astype(np.float32)
    w[1] = 0.0
    q, scales = quantize_int8_rows(w)
    assert scales[1] == 1.0
    assert np.all(q[1] == 0)
    assert np.all(dequantize_int8_rows(q, scales)[1] == 0.0)
    assert np.all(np.isfinite(scales))


def _pack(q: np.ndarray) -> np.ndarray:
    u = (q + 8).astype(np.uint8)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)


def test_int4_pack_unpack_roundtrip():
    rng = np.random.default_rng(3)
    q = rng.integers(-8, 8, size=(16, 256), dtype=np.int8)
    np.testing.assert_array_equal(unpack_int4(_pack(q)), q)


def test_int4_nibble_order():
    # Column 0 -> low nibble, column 1 -> high nibble, each stored as q+8.
    packed = np.array([[0x0F | (0x00 << 4), 0x08 | (0x09 << 4)]], dtype=np.uint8)
    np.testing.assert_array_equal(unpack_int4(packed), np.array([[7, -8, 0, 1]], dtype=np.int8))
    # And through the quantizer: a 1x128 row whose first four columns are hand-chosen.
    w = np.zeros((1, 128), dtype=np.float32)
    w[0, :4] = [7.0, -7.0, 0.0, 1.0]
    packed, scales = quantize_int4_groups(w)
    assert scales[0, 0] == np.float16(1.0)
    assert packed[0, 0] == (7 + 8) | ((-7 + 8) << 4)
    assert packed[0, 1] == (0 + 8) | ((1 + 8) << 4)


def test_int4_dequant_error_bound():
    rng = np.random.default_rng(4)
    w = rng.normal(0, 0.02, size=(32, 512)).astype(np.float32)
    packed, scales = quantize_int4_groups(w, 128)
    assert packed.dtype == np.uint8 and packed.shape == (32, 256)
    assert scales.dtype == np.float16 and scales.shape == (32, 4)
    deq = dequantize_int4_groups(packed, scales)
    bound = np.repeat(scales.astype(np.float32), 128, axis=1) / 2 + 1e-6
    assert np.all(np.abs(deq - w) <= bound)
