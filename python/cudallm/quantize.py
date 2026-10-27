"""Weight-only quantization used by the exporter and the reference model."""
import numpy as np


def quantize_int8_rows(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(w, dtype=np.float32)
    scales = np.abs(w).max(axis=1) / np.float32(127)
    scales = np.where(scales == 0, np.float32(1.0), scales).astype(np.float32)
    q = np.clip(np.rint(w / scales[:, None]), -127, 127).astype(np.int8)
    return q, scales


def dequantize_int8_rows(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scales.astype(np.float32)[:, None]


def quantize_int4_groups(w: np.ndarray, group: int = 128) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(w, dtype=np.float32)
    rows, cols = w.shape
    assert cols % group == 0 and cols % 2 == 0
    wg = w.reshape(rows, cols // group, group)
    scales = np.abs(wg).max(axis=2) / np.float32(7)
    scales = np.where(scales == 0, np.float32(1.0), scales)
    # Quantize against the fp16-rounded scale so dequantization uses exactly the stored value.
    scales_f16 = scales.astype(np.float16)
    q = np.clip(np.rint(wg / scales_f16.astype(np.float32)[:, :, None]), -8, 7).astype(np.int8)
    u = (q.reshape(rows, cols) + 8).astype(np.uint8)
    packed = (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)
    return packed, scales_f16


def unpack_int4(packed: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    rows, half = packed.shape
    out = np.empty((rows, half * 2), dtype=np.int8)
    out[:, 0::2] = (packed & 0x0F).astype(np.int8) - 8
    out[:, 1::2] = (packed >> 4).astype(np.int8) - 8
    return out


def dequantize_int4_groups(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    q = unpack_int4(packed).astype(np.float32)
    rows, cols = q.shape
    n_groups = scales.shape[1]
    group = cols // n_groups
    s = scales.astype(np.float32)
    return (q.reshape(rows, n_groups, group) * s[:, :, None]).reshape(rows, cols)
