"""Reader/writer for the .cudallm weight file format (DESIGN.md §7)."""
import struct
from pathlib import Path

import numpy as np

MAGIC = b"CLLM"
VERSION = 1

# Tensor dtype codes (§7.4).
F32 = 0
BF16 = 1
F16 = 2
I8 = 3
I4X2 = 4

# File-level weight_dtype codes (§7.2).
WEIGHT_BF16 = 1
WEIGHT_INT8 = 3
WEIGHT_INT4 = 4

HEADER_FMT = "<4s10I2f3IQ"
HEADER_SIZE = 256
ENTRY_FMT = "<48s4I2Q"
ENTRY_SIZE = 80
ALIGN = 256

HEADER_FIELDS = ("magic", "version", "weight_dtype", "hidden_size", "intermediate_size", "n_layers",
                 "n_heads", "n_kv_heads", "head_dim", "vocab_size", "max_seq_len", "rms_eps",
                 "rope_theta", "group_size", "n_tensors", "reserved0", "table_offset")

assert struct.calcsize(HEADER_FMT) == 72
assert struct.calcsize(ENTRY_FMT) == ENTRY_SIZE


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def expected_nbytes(dtype: int, rows: int, cols: int) -> int:
    if dtype == F32:
        return rows * cols * 4
    if dtype in (BF16, F16):
        return rows * cols * 2
    if dtype == I8:
        return rows * cols
    if dtype == I4X2:
        assert cols % 2 == 0
        return rows * cols // 2
    raise ValueError(f"unknown dtype code {dtype}")


def expected_names(n_layers: int, quant: bool) -> list[str]:
    names = ["rope.inv_freq", "tok_embeddings.weight"]
    if quant:
        names.append("tok_embeddings.scales")
    names.append("norm.weight")
    for i in range(n_layers):
        p = f"layers.{i}."
        names.append(p + "attn_norm.weight")
        for lin in ("wqkv", "wo"):
            names.append(p + lin + ".weight")
            if quant:
                names.append(p + lin + ".scales")
        names.append(p + "ffn_norm.weight")
        for lin in ("w_gate_up", "w_down"):
            names.append(p + lin + ".weight")
            if quant:
                names.append(p + lin + ".scales")
    return names


def write_cudallm(path: Path | str, header: dict, tensors: list[tuple[str, int, int, int, bytes]]) -> None:
    """Write a .cudallm file. `header` holds the model fields; magic, version, n_tensors and
    table_offset are filled in here."""
    n = len(tensors)
    entries = []
    offset = _align(HEADER_SIZE + ENTRY_SIZE * n)
    for name, dtype, rows, cols, data in tensors:
        nbytes = expected_nbytes(dtype, rows, cols)
        assert len(data) == nbytes, f"{name}: {len(data)} bytes, expected {nbytes}"
        name_b = name.encode("ascii")
        assert len(name_b) < 48, f"name too long: {name}"
        entries.append(struct.pack(ENTRY_FMT, name_b, dtype, rows, cols, 0, offset, nbytes))
        offset = _align(offset + nbytes)

    hdr = struct.pack(HEADER_FMT, MAGIC, VERSION, header["weight_dtype"], header["hidden_size"],
                      header["intermediate_size"], header["n_layers"], header["n_heads"],
                      header["n_kv_heads"], header["head_dim"], header["vocab_size"],
                      header["max_seq_len"], header["rms_eps"], header["rope_theta"],
                      header["group_size"], n, 0, HEADER_SIZE)
    hdr += b"\0" * (HEADER_SIZE - len(hdr))

    with open(path, "wb") as f:
        f.write(hdr)
        for e in entries:
            f.write(e)
        # Zero padding before each tensor only; the file ends at last offset + nbytes.
        for _, _, _, _, data in tensors:
            f.write(b"\0" * (_align(f.tell()) - f.tell()))
            f.write(data)


def read_header(path: Path | str) -> dict:
    with open(path, "rb") as f:
        raw = f.read(HEADER_SIZE)
    if len(raw) != HEADER_SIZE:
        raise ValueError(f"{path}: truncated header")
    vals = struct.unpack(HEADER_FMT, raw[:struct.calcsize(HEADER_FMT)])
    return dict(zip(HEADER_FIELDS, vals))


def read_table(path: Path | str) -> list[dict]:
    hdr = read_header(path)
    n = hdr["n_tensors"]
    with open(path, "rb") as f:
        f.seek(hdr["table_offset"])
        raw = f.read(ENTRY_SIZE * n)
    out = []
    for k in range(n):
        name, dtype, rows, cols, reserved, offset, nbytes = struct.unpack_from(ENTRY_FMT, raw, k * ENTRY_SIZE)
        out.append({"name": name.rstrip(b"\0").decode("ascii"), "dtype": dtype, "rows": rows,
                    "cols": cols, "reserved": reserved, "offset": offset, "nbytes": nbytes})
    return out


def load_tensor(path: Path | str, entry: dict) -> np.ndarray:
    rows, cols, dtype = entry["rows"], entry["cols"], entry["dtype"]
    np_dtype, shape = {
        F32: (np.float32, (rows, cols)),
        BF16: (np.uint16, (rows, cols)),
        F16: (np.float16, (rows, cols)),
        I8: (np.int8, (rows, cols)),
        I4X2: (np.uint8, (rows, cols // 2)),
    }[dtype]
    count = shape[0] * shape[1]
    arr = np.fromfile(path, dtype=np_dtype, count=count, offset=entry["offset"])
    if arr.size != count:
        raise ValueError(f"{path}: truncated tensor {entry['name']}")
    arr = arr.reshape(shape)
    if dtype == F32 and cols == 1:
        arr = arr.reshape(rows)
    return arr


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)
