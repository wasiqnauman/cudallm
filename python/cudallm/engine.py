"""ctypes wrapper around build/libcudallm.so (DESIGN.md §12.7)."""
import ctypes
import functools
from ctypes import POINTER, c_char_p, c_float, c_int, c_ulonglong, c_ushort, c_void_p
from pathlib import Path

import numpy as np

from .paths import LIB_PATH

_IMPL = {"naive": 0, "opt": 1}
_WTYPE = {"bf16": 1, "int8": 3, "int4": 4}
_EPILOGUE = {"store": 0, "add": 1, "swiglu": 2}


class CudallmOptions(ctypes.Structure):
    _fields_ = [("matvec_impl", c_int), ("attn_impl", c_int), ("fuse", c_int), ("cuda_graph", c_int),
                ("max_seq_len", c_int), ("matvec_rpb", c_int)]


class CudallmConfigInfo(ctypes.Structure):
    _fields_ = [("weight_dtype", c_int), ("hidden", c_int), ("inter", c_int), ("n_layers", c_int),
                ("n_heads", c_int), ("n_kv_heads", c_int), ("head_dim", c_int), ("vocab", c_int),
                ("max_seq_len", c_int), ("group_size", c_int), ("rms_eps", c_float),
                ("weight_bytes", c_ulonglong)]


class CudaLLMError(RuntimeError):
    def __init__(self, code: int, message: str):
        super().__init__(f"cudallm error {code}: {message}")
        self.code = code


_FLOAT_P = POINTER(c_float)
_INT_P = POINTER(c_int)
_USHORT_P = POINTER(c_ushort)


@functools.cache
def _lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(LIB_PATH))
    sigs = {
        "cudallm_last_error": ([], c_char_p),
        "cudallm_default_options": ([POINTER(CudallmOptions)], None),
        "cudallm_create": ([c_char_p, POINTER(CudallmOptions), POINTER(c_void_p)], c_int),
        "cudallm_destroy": ([c_void_p], None),
        "cudallm_get_config": ([c_void_p, POINTER(CudallmConfigInfo)], c_int),
        "cudallm_reset": ([c_void_p], c_int),
        "cudallm_pos": ([c_void_p], c_int),
        "cudallm_step": ([c_void_p, c_int, _FLOAT_P, _INT_P], c_int),
        "cudallm_step_debug": ([c_void_p, c_int, _FLOAT_P, _FLOAT_P, _INT_P], c_int),
        "cudallm_kernel_matvec": ([c_int, c_int, c_int, c_int, c_int, c_void_p, c_ulonglong, c_void_p, c_ulonglong,
                                   _FLOAT_P, _FLOAT_P, c_float, _FLOAT_P, c_int, c_int], c_int),
        "cudallm_kernel_rmsnorm": ([_FLOAT_P, _FLOAT_P, c_float, _FLOAT_P, c_int], c_int),
        "cudallm_kernel_rope_kv": ([c_int, _FLOAT_P, _FLOAT_P, c_int, _USHORT_P, _USHORT_P, c_int, c_int, c_int, c_int],
                                   c_int),
        "cudallm_kernel_attention": ([c_int, _FLOAT_P, _USHORT_P, _USHORT_P, c_int, _FLOAT_P, c_int, c_int, c_int, c_int],
                                     c_int),
        "cudallm_kernel_silu_mul": ([_FLOAT_P, _FLOAT_P, c_int], c_int),
        "cudallm_kernel_argmax": ([_FLOAT_P, c_int, _INT_P], c_int),
    }
    for name, (argtypes, restype) in sigs.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype
    return lib


def _check(rc: int) -> None:
    if rc != 0:
        msg = _lib().cudallm_last_error()
        raise CudaLLMError(rc, msg.decode("utf-8", "replace") if msg else "")


def _fp(a: np.ndarray):
    return a.ctypes.data_as(_FLOAT_P)


def _f32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


def _fp16_bits(a) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype == np.float16:
        return np.ascontiguousarray(a).view(np.uint16).copy()
    return np.ascontiguousarray(a, dtype=np.uint16).copy()


def _code(table: dict[str, int], v: str | int) -> int:
    return table[v] if isinstance(v, str) else int(v)


class Engine:
    def __init__(self, path: Path | str, *, matvec: str = "opt", attn: str = "opt", fuse: bool = True,
                 cuda_graph: bool = True, max_seq_len: int = 0, matvec_rpb: int = 0):
        self._h = None
        lib = _lib()
        opts = CudallmOptions(_IMPL[matvec], _IMPL[attn], int(bool(fuse)), int(bool(cuda_graph)), int(max_seq_len),
                              int(matvec_rpb))
        h = c_void_p()
        _check(lib.cudallm_create(str(path).encode(), ctypes.byref(opts), ctypes.byref(h)))
        self._h = h
        info = CudallmConfigInfo()
        _check(lib.cudallm_get_config(self._h, ctypes.byref(info)))
        self.config: dict = {name: getattr(info, name) for name, _ in CudallmConfigInfo._fields_}
        self._vocab = info.vocab
        self._rows = info.n_layers + 1
        self._hidden = info.hidden

    def _handle(self) -> c_void_p:
        if self._h is None:
            raise CudaLLMError(5, "engine is closed")
        return self._h

    @property
    def pos(self) -> int:
        return _lib().cudallm_pos(self._handle())

    def reset(self) -> None:
        _check(_lib().cudallm_reset(self._handle()))

    def step(self, token: int | None = None, *, want_logits: bool = False) -> tuple[int, np.ndarray | None]:
        nxt = c_int(0)
        logits = np.empty(self._vocab, dtype=np.float32) if want_logits else None
        _check(_lib().cudallm_step(self._handle(), -1 if token is None else int(token),
                                   _fp(logits) if logits is not None else None, ctypes.byref(nxt)))
        return nxt.value, logits

    def step_debug(self, token: int | None = None) -> tuple[int, np.ndarray, np.ndarray]:
        nxt = c_int(0)
        logits = np.empty(self._vocab, dtype=np.float32)
        residuals = np.empty((self._rows, self._hidden), dtype=np.float32)
        _check(_lib().cudallm_step_debug(self._handle(), -1 if token is None else int(token), _fp(residuals),
                                         _fp(logits), ctypes.byref(nxt)))
        return nxt.value, logits, residuals

    def close(self) -> None:
        if self._h is not None:
            _lib().cudallm_destroy(self._h)
            self._h = None

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_h", None) is not None:
            self.close()


def kernel_matvec(impl: str | int, wtype: str | int, epilogue: str | int, fuse_norm: bool | int, rpb: int,
                  w: np.ndarray, scales: np.ndarray | None, x: np.ndarray, norm_w: np.ndarray | None, eps: float,
                  y_init: np.ndarray | None, rows: int, cols: int) -> np.ndarray:
    ep = _code(_EPILOGUE, epilogue)
    out_rows = rows // 2 if ep == 2 else rows
    w = np.ascontiguousarray(w)
    s = np.ascontiguousarray(scales) if scales is not None else None
    x = _f32(x)
    n = _f32(norm_w) if norm_w is not None else None
    y = _f32(y_init).copy() if y_init is not None else np.zeros(out_rows, dtype=np.float32)
    assert y.shape == (out_rows,)
    _check(_lib().cudallm_kernel_matvec(_code(_IMPL, impl), _code(_WTYPE, wtype), ep, int(bool(fuse_norm)), int(rpb),
                                        w.ctypes.data, w.nbytes, s.ctypes.data if s is not None else None,
                                        s.nbytes if s is not None else 0, _fp(x), _fp(n) if n is not None else None,
                                        float(eps), _fp(y), int(rows), int(cols)))
    return y


def kernel_rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    x, w = _f32(x), _f32(w)
    y = np.empty_like(x)
    _check(_lib().cudallm_kernel_rmsnorm(_fp(x), _fp(w), float(eps), _fp(y), x.size))
    return y


def kernel_rope_kv(fused: bool | int, qkv: np.ndarray, inv_freq: np.ndarray, pos: int, k_cache: np.ndarray,
                   v_cache: np.ndarray, nh: int, nkv: int, hd: int,
                   max_seq: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qkv = _f32(qkv).copy()
    inv = _f32(inv_freq)
    k = _fp16_bits(k_cache)
    v = _fp16_bits(v_cache)
    _check(_lib().cudallm_kernel_rope_kv(int(bool(fused)), _fp(qkv), _fp(inv), int(pos), k.ctypes.data_as(_USHORT_P),
                                         v.ctypes.data_as(_USHORT_P), nh, nkv, hd, max_seq))
    return qkv, k.view(np.float16), v.view(np.float16)


def kernel_attention(impl: str | int, q: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray, ctx: int, nh: int,
                     nkv: int, hd: int, max_seq: int) -> np.ndarray:
    q = _f32(q)
    k = _fp16_bits(k_cache)
    v = _fp16_bits(v_cache)
    out = np.empty(nh * hd, dtype=np.float32)
    _check(_lib().cudallm_kernel_attention(_code(_IMPL, impl), _fp(q), k.ctypes.data_as(_USHORT_P),
                                           v.ctypes.data_as(_USHORT_P), int(ctx), _fp(out), nh, nkv, hd, max_seq))
    return out


def kernel_silu_mul(gu: np.ndarray) -> np.ndarray:
    gu = _f32(gu)
    inter = gu.size // 2
    act = np.empty(inter, dtype=np.float32)
    _check(_lib().cudallm_kernel_silu_mul(_fp(gu), _fp(act), inter))
    return act


def kernel_argmax(logits: np.ndarray) -> int:
    logits = _f32(logits)
    idx = c_int(-1)
    _check(_lib().cudallm_kernel_argmax(_fp(logits), logits.size, ctypes.byref(idx)))
    return idx.value
