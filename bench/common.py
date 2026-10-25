"""Shared helpers for the benchmark scripts (DESIGN.md §15.1)."""
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from cudallm.paths import BUILD_DIR, MODEL_PATHS, RESULTS_DIR  # noqa: E402

__all__ = ["BUILD_DIR", "MODEL_PATHS", "REPO_ROOT", "RESULTS_DIR", "git_commit", "gpu_snapshot", "load_peak_gbps",
           "run", "save_json"]


def run(cmd: list[str]) -> str:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    res = subprocess.run([str(c) for c in cmd], check=True, capture_output=True, text=True, cwd=REPO_ROOT)
    sys.stdout.write(res.stdout)
    sys.stderr.write(res.stderr)
    return res.stdout


def git_commit() -> str:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True,
                            cwd=REPO_ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True,
                           cwd=REPO_ROOT).stdout.strip()
    return commit + ("-dirty" if dirty else "")


def gpu_snapshot() -> dict:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw",
                          "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout
    name, sm, mem, temp, power = [f.strip() for f in out.strip().splitlines()[0].split(",")]

    def num(v: str) -> float | None:
        try:
            return float(v)
        except ValueError:
            return None
    return {"name": name, "clocks_sm_mhz": num(sm), "clocks_mem_mhz": num(mem), "temperature_c": num(temp),
            "power_draw_w": num(power)}


def save_json(path: Path | str, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    print(f"saved {path}")


def load_peak_gbps() -> float:
    path = RESULTS_DIR / "bandwidth.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run bench/bench_bandwidth.py first")
    with open(path) as f:
        return float(json.load(f)["read_gbps_best"])
