"""Pure-PyTorch CPU reference that reads a .cudallm file.

Weights stay in their stored form and are materialized to fp32 per matmul, which mirrors the
engine's math while keeping the resident set at roughly the file size.
"""
import os
from pathlib import Path

import numpy as np
import torch

from . import format as fmt


class _Linear:
    """One stored linear weight; `matvec` materializes fp32 and computes W @ x."""

    def __init__(self, path: Path, weight: dict, scales: dict | None):
        self.rows, self.cols, self.dtype = weight["rows"], weight["cols"], weight["dtype"]
        w = fmt.load_tensor(path, weight)
        if self.dtype == fmt.BF16:
            self.w = torch.from_numpy(w.view(np.int16)).view(torch.bfloat16)
            self.scales = None
        elif self.dtype == fmt.I8:
            self.w = torch.from_numpy(w)
            self.scales = torch.from_numpy(fmt.load_tensor(path, scales).astype(np.float32))
        elif self.dtype == fmt.I4X2:
            self.w = torch.from_numpy(w)
            self.scales = torch.from_numpy(fmt.load_tensor(path, scales))
        else:
            raise ValueError(f"unsupported linear dtype {self.dtype}")

    def _dequant(self, w: torch.Tensor, scales: torch.Tensor | None) -> torch.Tensor:
        if self.dtype == fmt.BF16:
            return w.float()
        if self.dtype == fmt.I8:
            return w.float() * scales[:, None]
        # I4X2: byte b holds element 2b (low nibble) and 2b+1 (high nibble), stored as q+8.
        rows = w.shape[0]
        q = torch.empty((rows, self.cols), dtype=torch.float32)
        q[:, 0::2] = (w & 0x0F).to(torch.float32) - 8.0
        q[:, 1::2] = (w >> 4).to(torch.float32) - 8.0
        n_groups = scales.shape[1]
        s = scales.to(torch.float32)
        return (q.view(rows, n_groups, self.cols // n_groups) * s[:, :, None]).view(rows, self.cols)

    def dense(self) -> torch.Tensor:
        return self._dequant(self.w, self.scales)

    def row(self, r: int) -> torch.Tensor:
        s = None if self.scales is None else self.scales[r:r + 1]
        return self._dequant(self.w[r:r + 1], s)[0]

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense() @ x


def _rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(torch.mean(x * x) + eps) * w


def _silu(g: torch.Tensor) -> torch.Tensor:
    return g / (1.0 + torch.exp(-g))


class ReferenceModel:
    def __init__(self, path: Path, kv_dtype: torch.dtype = torch.float16, device: str = "cpu"):
        assert device == "cpu", "the reference runs on CPU only"
        torch.set_num_threads(os.cpu_count())
        self.path = Path(path)
        self.kv_dtype = kv_dtype
        hdr = fmt.read_header(self.path)
        self.hidden = hdr["hidden_size"]
        self.inter = hdr["intermediate_size"]
        self.n_layers = hdr["n_layers"]
        self.n_heads = hdr["n_heads"]
        self.n_kv_heads = hdr["n_kv_heads"]
        self.head_dim = hdr["head_dim"]
        self.vocab = hdr["vocab_size"]
        self.max_seq_len = hdr["max_seq_len"]
        self.eps = hdr["rms_eps"]

        quant = hdr["weight_dtype"] != fmt.WEIGHT_BF16
        table = {e["name"]: e for e in fmt.read_table(self.path)}
        assert list(table) == fmt.expected_names(self.n_layers, quant), "unexpected tensor names/order"

        def lin(name: str) -> _Linear:
            return _Linear(self.path, table[name + ".weight"], table.get(name + ".scales"))

        def vec(name: str) -> torch.Tensor:
            return torch.from_numpy(fmt.load_tensor(self.path, table[name]).astype(np.float32))

        self.inv_freq = vec("rope.inv_freq")
        self.tok_embeddings = lin("tok_embeddings")
        self.final_norm = vec("norm.weight")
        self.layers = []
        for i in range(self.n_layers):
            p = f"layers.{i}."
            self.layers.append({
                "attn_norm": vec(p + "attn_norm.weight"),
                "wqkv": lin(p + "wqkv"),
                "wo": lin(p + "wo"),
                "ffn_norm": vec(p + "ffn_norm.weight"),
                "w_gate_up": lin(p + "w_gate_up"),
                "w_down": lin(p + "w_down"),
            })
        shape = (self.n_kv_heads, self.max_seq_len, self.head_dim)
        self.k_cache = [torch.zeros(shape, dtype=kv_dtype) for _ in range(self.n_layers)]
        self.v_cache = [torch.zeros(shape, dtype=kv_dtype) for _ in range(self.n_layers)]
        self.pos = 0

    def reset(self) -> None:
        self.pos = 0

    def _rope(self, v: torch.Tensor) -> torch.Tensor:
        # v: [n, head_dim]; pairs (i, i+32), HF rotate_half convention.
        half = self.head_dim // 2
        ang = torch.tensor(self.pos, dtype=torch.float32) * self.inv_freq
        c, s = torch.cos(ang), torch.sin(ang)
        a, b = v[:, :half], v[:, half:]
        return torch.cat([a * c - b * s, b * c + a * s], dim=1)

    @torch.no_grad()
    def step(self, token: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Process token at self.pos. Returns (logits[vocab] f32, residuals: list of n_layers+1 tensors [hidden])
        where residuals[0] = embedding output and residuals[l+1] = residual after layer l."""
        assert self.pos < self.max_seq_len
        H, NH, KV, D = self.hidden, self.n_heads, self.n_kv_heads, self.head_dim
        group = NH // KV
        pos = self.pos
        x = self.tok_embeddings.row(token).clone()
        residuals = [x.clone()]
        for l, L in enumerate(self.layers):
            xb = _rmsnorm(x, L["attn_norm"], self.eps)
            qkv = L["wqkv"].matvec(xb)
            q = self._rope(qkv[:NH * D].view(NH, D))
            k = self._rope(qkv[NH * D:(NH + KV) * D].view(KV, D))
            v = qkv[(NH + KV) * D:].view(KV, D)
            self.k_cache[l][:, pos] = k.to(self.kv_dtype)
            self.v_cache[l][:, pos] = v.to(self.kv_dtype)
            K = self.k_cache[l][:, :pos + 1].float()  # [KV, ctx, D]
            V = self.v_cache[l][:, :pos + 1].float()
            att = torch.empty(NH, D, dtype=torch.float32)
            for h in range(NH):
                kvh = h // group
                s = (K[kvh] @ q[h]) * 0.125
                p = torch.exp(s - s.max())
                p = p / p.sum()
                att[h] = p @ V[kvh]
            x = x + L["wo"].matvec(att.view(H))
            xb = _rmsnorm(x, L["ffn_norm"], self.eps)
            gu = L["w_gate_up"].matvec(xb)
            act = _silu(gu[:self.inter]) * gu[self.inter:]
            x = x + L["w_down"].matvec(act)
            residuals.append(x.clone())
        logits = self.tok_embeddings.matvec(_rmsnorm(x, self.final_norm, self.eps))
        self.pos += 1
        return logits, residuals

    def greedy(self, prompt_ids: list[int], n_new: int) -> list[int]:
        self.reset()
        logits = None
        for t in prompt_ids:
            logits, _ = self.step(t)
        out: list[int] = []
        while len(out) < n_new:
            nxt = int(torch.argmax(logits))
            out.append(nxt)
            if len(out) < n_new:
                logits, _ = self.step(nxt)
        return out
