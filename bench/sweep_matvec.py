"""Rows-per-block sweep for the opt matvec.

Runs cudallm_kernelbench --op shapes for each weight type and rpb, picks per weight type the rpb with
the lowest weighted_us (ties -> smaller rpb), saves results/sweep_matvec.json and prints the winners.
"""
import json
import tempfile
from pathlib import Path

from common import BUILD_DIR, RESULTS_DIR, git_commit, gpu_snapshot, run, save_json

WTYPES = ("bf16", "int8", "int4")
RPBS = (1, 2, 4, 8)


def main() -> None:
    gpu_before = gpu_snapshot()
    data: dict[str, dict[str, dict]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for wt in WTYPES:
            data[wt] = {}
            for rpb in RPBS:
                out = Path(tmp) / f"{wt}_{rpb}.json"
                run([BUILD_DIR / "cudallm_kernelbench", "--op", "shapes", "--wtype", wt, "--rpb", rpb, "--json", out])
                with open(out) as f:
                    data[wt][str(rpb)] = json.load(f)
    winners = {}
    for wt in WTYPES:
        # Ties go to the smaller rpb because RPBS is ascending and min() keeps the first minimum.
        winners[wt] = min(RPBS, key=lambda r: data[wt][str(r)]["weighted_us"])
    save_json(RESULTS_DIR / "sweep_matvec.json", {
        "tool": "sweep_matvec", "results": data, "winners": winners,
        "weighted_us": {wt: {str(r): data[wt][str(r)]["weighted_us"] for r in RPBS} for wt in WTYPES},
        "git_commit": git_commit(), "gpu_before": gpu_before, "gpu_after": gpu_snapshot(),
    })
    for wt in WTYPES:
        print(f"winner {wt}: rpb={winners[wt]} weighted_us={data[wt][str(winners[wt])]['weighted_us']:.2f}")


if __name__ == "__main__":
    main()
