"""Draw the README figures (light + dark variants) into docs/img/readme/.

Numbers are the ones in docs/RESULTS.md; update them here if the report changes.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "docs" / "img" / "readme"

THEMES = {
    "light": dict(bg="#ffffff", ink="#1f2328", muted="#656d76", grid="#d8dee4",
                  cudallm="#2a78d6", llama="#eb6834", hf="#1baf7a", roof="#656d76",
                  neutral="#afb8c1"),
    "dark": dict(bg="#0d1117", ink="#e6edf3", muted="#9198a1", grid="#30363d",
                 cudallm="#3987e5", llama="#d95926", hf="#199e70", roof="#9198a1",
                 neutral="#484f58"),
}

# tg256 decode, tok/s (RESULTS.md section 7)
COMPARE = {
    "16-bit": {"cudallm": 124.68, "llama.cpp": 123.07, "HF transformers": 90.22},
    "8-bit": {"cudallm": 231.22, "llama.cpp": 215.60},
    "4-bit": {"cudallm": 381.18},
}
ABLATION = [  # (label, tok/s)
    ("Naive kernels", 18.88),
    ("+ coalesced matvec", 115.75),
    ("+ split-K GQA attention", 119.83),
    ("+ kernel fusion", 123.80),
    ("+ CUDA graph", 124.80),
]
ROOFLINE = 138.41
CONTEXT = [128, 512, 1024, 2048, 4000]
CTX_OPT = [123.65, 122.90, 122.65, 119.72, 116.59]
CTX_NAIVE = [121.05, 113.19, 100.30, 84.77, 64.90]
QUANT = [  # (dtype, tok/s, perplexity)
    ("bf16", 123.59, 18.4914),
    ("int8", 229.75, 18.5731),
    ("int4", 383.24, 24.4830),
]


def style(ax, t):
    ax.set_facecolor(t["bg"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=10)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def new_fig(t, w, h, **kw):
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": t["ink"]})
    fig, ax = plt.subplots(figsize=(w, h), facecolor=t["bg"], **kw)
    return fig, ax


def save(fig, name, mode):
    fig.savefig(OUT / f"{name}-{mode}.png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def compare(t, mode):
    fig, ax = new_fig(t, 8, 4)
    style(ax, t)
    engines = [("cudallm", t["cudallm"]), ("llama.cpp", t["llama"]),
               ("HF transformers", t["hf"])]
    width = 0.26
    for gi, (group, vals) in enumerate(COMPARE.items()):
        present = [(e, c) for e, c in engines if e in vals]
        for ei, (eng, color) in enumerate(present):
            x = gi + (ei - (len(present) - 1) / 2) * width
            v = vals[eng]
            ax.bar(x, v, width * 0.92, color=color, label=eng if gi == 0 else None)
            ax.text(x, v + 6, f"{v:.0f}", ha="center", va="bottom", fontsize=10,
                    color=t["ink"], fontweight="bold" if eng == "cudallm" else "normal")
    ax.set_xticks(range(len(COMPARE)))
    ax.set_xticklabels([f"{g} weights" for g in COMPARE], fontsize=11, color=t["ink"])
    ax.set_ylabel("tokens / s", color=t["muted"])
    ax.set_ylim(0, 420)
    leg = ax.legend(frameon=False, loc="upper left", fontsize=10)
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    save(fig, "compare", mode)


def ablation(t, mode):
    fig, ax = new_fig(t, 8, 3.4)
    style(ax, t)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    labels = [a for a, _ in ABLATION][::-1]
    vals = [v for _, v in ABLATION][::-1]
    colors = [t["cudallm"]] + [t["neutral"]] * (len(vals) - 1)
    ax.barh(labels, vals, height=0.62, color=colors)
    for y, v in enumerate(vals):
        ax.text(v + 2, y, f"{v:.1f}", va="center", fontsize=10, color=t["ink"])
    ax.axvline(ROOFLINE, color=t["roof"], linestyle="--", linewidth=1.2)
    ax.text(ROOFLINE + 2, len(vals) - 0.5, f"memory roofline\n{ROOFLINE:.0f} tok/s",
            fontsize=9, color=t["muted"], va="top")
    ax.set_xlim(0, 165)
    ax.set_xlabel("bf16 decode, tokens / s", color=t["muted"])
    ax.tick_params(axis="y", labelcolor=t["ink"], labelsize=10)
    ax.spines["bottom"].set_visible(False)
    save(fig, "ablation", mode)


def context(t, mode):
    fig, ax = new_fig(t, 8, 3.6)
    style(ax, t)
    ax.plot(CONTEXT, CTX_OPT, color=t["cudallm"], linewidth=2.2, marker="o", markersize=6)
    ax.plot(CONTEXT, CTX_NAIVE, color=t["neutral"], linewidth=2.2, marker="o", markersize=6)
    ax.text(CONTEXT[-1] + 80, CTX_OPT[-1], "split-K attention", va="center",
            fontsize=10, color=t["ink"])
    ax.text(CONTEXT[-1] + 80, CTX_NAIVE[-1], "naive attention", va="center",
            fontsize=10, color=t["muted"])
    ax.vlines(CONTEXT[-1], CTX_NAIVE[-1] + 4, CTX_OPT[-1] - 4, color=t["muted"], linewidth=1)
    ax.text(CONTEXT[-1] - 120, (CTX_OPT[-1] + CTX_NAIVE[-1]) / 2, "1.8× faster",
            fontsize=11, fontweight="bold", color=t["ink"], va="center", ha="right")
    ax.set_xlim(0, 5200)
    ax.set_ylim(0, 140)
    ax.set_xticks(CONTEXT[1:] + [0])
    ax.set_xlabel("context length (tokens)", color=t["muted"])
    ax.set_ylabel("tokens / s", color=t["muted"])
    save(fig, "context", mode)


def quant(t, mode):
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": t["ink"]})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2), facecolor=t["bg"])
    names = [q[0] for q in QUANT]
    for ax, idx, title, fmt, top in ((a1, 1, "Speed (tokens / s)", "{:.0f}", 450),
                                     (a2, 2, "WikiText-2 perplexity (lower is better)", "{:.1f}", 29)):
        style(ax, t)
        vals = [q[idx] for q in QUANT]
        ax.bar(names, vals, width=0.55, color=t["cudallm"])
        for x, v in enumerate(vals):
            ax.text(x, v + top * 0.015, fmt.format(v), ha="center", va="bottom",
                    fontsize=10, color=t["ink"])
        ax.set_title(title, fontsize=11, color=t["ink"], loc="left")
        ax.set_ylim(0, top)
        ax.tick_params(axis="x", labelcolor=t["ink"], labelsize=11)
    fig.tight_layout(w_pad=3)
    save(fig, "quant", mode)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, t in THEMES.items():
        compare(t, mode)
        ablation(t, mode)
        context(t, mode)
        quant(t, mode)
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
