"""Measure DRAM bandwidth with build/cudallm_bandwidth."""
import json

from common import BUILD_DIR, RESULTS_DIR, git_commit, gpu_snapshot, run, save_json


def main() -> None:
    out = RESULTS_DIR / "bandwidth.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    run([BUILD_DIR / "cudallm_bandwidth", "--json", out])
    with open(out) as f:
        data = json.load(f)
    data["git_commit"] = git_commit()
    data["gpu_snapshot"] = gpu_snapshot()
    save_json(out, data)


if __name__ == "__main__":
    main()
