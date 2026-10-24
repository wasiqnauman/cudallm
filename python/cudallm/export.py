"""Convert the HF Llama-3.2-1B-Instruct checkpoint to a .cudallm file (DESIGN.md §12.4).

Usage: python -m cudallm.export --hf-dir DIR --out FILE --dtype {bf16,int8,int4}
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from . import format as fmt
from .quantize import quantize_int4_groups, quantize_int8_rows

MAX_SEQ_LEN = 4096
INT4_GROUP = 128

EXPECTED_CONFIG = {
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "vocab_size": 128256,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "tie_word_embeddings": True,
    "bos_token_id": 128000,
}
EXPECTED_ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}
EXPECTED_EOS = [128001, 128008, 128009]


def check_config(config) -> None:
    for key, want in EXPECTED_CONFIG.items():
        got = getattr(config, key)
        assert got == want, f"config.{key} = {got!r}, expected {want!r}"
    rs = config.rope_scaling
    for key, want in EXPECTED_ROPE_SCALING.items():
        assert rs.get(key) == want, f"config.rope_scaling[{key!r}] = {rs.get(key)!r}, expected {want!r}"
    assert list(config.eos_token_id) == EXPECTED_EOS, f"config.eos_token_id = {config.eos_token_id!r}"
    assert config.torch_dtype == torch.bfloat16, f"config.torch_dtype = {config.torch_dtype!r}"


def compute_inv_freq(config) -> np.ndarray:
    inv_freq, attn_scaling = ROPE_INIT_FUNCTIONS["llama3"](config, "cpu")
    assert attn_scaling == 1.0, f"attention scaling {attn_scaling} != 1.0"
    assert tuple(inv_freq.shape) == (32,), f"inv_freq shape {tuple(inv_freq.shape)}"
    return inv_freq.to(torch.float32).numpy()


def hf_layer_keys(i: int) -> dict[str, str]:
    p = f"model.layers.{i}."
    return {
        "attn_norm": p + "input_layernorm.weight",
        "q": p + "self_attn.q_proj.weight",
        "k": p + "self_attn.k_proj.weight",
        "v": p + "self_attn.v_proj.weight",
        "o": p + "self_attn.o_proj.weight",
        "ffn_norm": p + "post_attention_layernorm.weight",
        "gate": p + "mlp.gate_proj.weight",
        "up": p + "mlp.up_proj.weight",
        "down": p + "mlp.down_proj.weight",
    }


def expected_hf_keys(n_layers: int) -> list[str]:
    keys = ["model.embed_tokens.weight", "model.norm.weight"]
    for i in range(n_layers):
        keys.extend(hf_layer_keys(i).values())
    return keys


def linear_tensors(name: str, t: torch.Tensor, dtype: str) -> list[tuple[str, int, int, int, bytes]]:
    assert t.dtype == torch.bfloat16, f"{name}: dtype {t.dtype}"
    rows, cols = t.shape
    if dtype == "bf16":
        return [(name + ".weight", fmt.BF16, rows, cols, t.contiguous().view(torch.int16).numpy().tobytes())]
    w = t.float().numpy()
    if dtype == "int8":
        q, scales = quantize_int8_rows(w)
        return [(name + ".weight", fmt.I8, rows, cols, q.tobytes()),
                (name + ".scales", fmt.F32, rows, 1, scales.astype(np.float32).tobytes())]
    packed, scales = quantize_int4_groups(w, INT4_GROUP)
    return [(name + ".weight", fmt.I4X2, rows, cols, packed.tobytes()),
            (name + ".scales", fmt.F16, rows, cols // INT4_GROUP, scales.astype(np.float16).tobytes())]


def norm_tensor(name: str, t: torch.Tensor) -> tuple[str, int, int, int, bytes]:
    a = t.float().numpy().astype(np.float32)
    return (name, fmt.F32, a.shape[0], 1, a.tobytes())


def export(hf_dir: Path, out: Path, dtype: str) -> None:
    config = AutoConfig.from_pretrained(hf_dir)
    check_config(config)
    n_layers = config.num_hidden_layers
    inv_freq = compute_inv_freq(config)

    sd: dict[str, torch.Tensor] = {}
    files = sorted(hf_dir.glob("*.safetensors"))
    assert files, f"no *.safetensors in {hf_dir}"
    for f in files:
        sd.update(load_file(str(f)))
    assert "lm_head.weight" not in sd, "unexpected lm_head.weight (embeddings must be tied)"
    missing = [k for k in expected_hf_keys(n_layers) if k not in sd]
    assert not missing, f"missing HF tensors: {missing[:5]}"

    # Tensors are popped as they are converted so the bf16 originals are freed layer by layer.
    tensors = [("rope.inv_freq", fmt.F32, 32, 1, inv_freq.tobytes())]
    tensors += linear_tensors("tok_embeddings", sd.pop("model.embed_tokens.weight"), dtype)
    tensors.append(norm_tensor("norm.weight", sd.pop("model.norm.weight")))
    for i in range(n_layers):
        k = hf_layer_keys(i)
        p = f"layers.{i}."
        tensors.append(norm_tensor(p + "attn_norm.weight", sd.pop(k["attn_norm"])))
        wqkv = torch.cat([sd.pop(k["q"]), sd.pop(k["k"]), sd.pop(k["v"])], dim=0)
        tensors += linear_tensors(p + "wqkv", wqkv, dtype)
        del wqkv
        tensors += linear_tensors(p + "wo", sd.pop(k["o"]), dtype)
        tensors.append(norm_tensor(p + "ffn_norm.weight", sd.pop(k["ffn_norm"])))
        w_gate_up = torch.cat([sd.pop(k["gate"]), sd.pop(k["up"])], dim=0)
        tensors += linear_tensors(p + "w_gate_up", w_gate_up, dtype)
        del w_gate_up
        tensors += linear_tensors(p + "w_down", sd.pop(k["down"]), dtype)

    quant = dtype != "bf16"
    names = [t[0] for t in tensors]
    assert names == fmt.expected_names(n_layers, quant), "tensor order does not match §7.5"

    weight_dtype = {"bf16": fmt.WEIGHT_BF16, "int8": fmt.WEIGHT_INT8, "int4": fmt.WEIGHT_INT4}[dtype]
    header = {
        "weight_dtype": weight_dtype,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "n_layers": n_layers,
        "n_heads": config.num_attention_heads,
        "n_kv_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "vocab_size": config.vocab_size,
        "max_seq_len": MAX_SEQ_LEN,
        "rms_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "group_size": INT4_GROUP if dtype == "int4" else 0,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt.write_cudallm(out, header, tensors)
    print(f"wrote {out}: {os.path.getsize(out)} bytes, {len(tensors)} tensors")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export HF Llama-3.2-1B-Instruct to .cudallm")
    ap.add_argument("--hf-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", choices=("bf16", "int8", "int4"), required=True)
    args = ap.parse_args(argv)
    export(args.hf_dir, args.out, args.dtype)
    return 0


if __name__ == "__main__":
    sys.exit(main())
