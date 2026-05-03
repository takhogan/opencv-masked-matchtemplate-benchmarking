"""Bar plots of average runtime by system, split by synthetic vs real images."""

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")


GPU_BACKENDS = {"umat", "cuda"}


def collect_runs(payload, kind, group):
    runs = []
    for build in payload.values():
        if not isinstance(build, dict):
            continue
        for sc in build.get("scenarios", []):
            if sc.get("spec", {}).get("kind") != kind:
                continue
            for method_res in sc.get("results", {}).values():
                for backend, backend_res in method_res.items():
                    if not backend_res:
                        continue
                    is_gpu = backend in GPU_BACKENDS
                    if (group == "gpu") != is_gpu:
                        continue
                    runs.extend(backend_res.get("runs", []))
    return np.array(runs, dtype=float)


def main():
    payloads = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_bench_results.json"))):
        name = os.path.basename(path).split("_bench_results.json")[0]
        with open(path) as fh:
            payloads[name] = json.load(fh)

    kinds = [("synthetic", "synthetic"), ("files", "real")]
    groups = [("cpu", "CPU"), ("gpu", "GPU")]
    columns = [(s, kind, klabel) for s in payloads for kind, klabel in kinds]
    x = np.arange(len(columns))
    width = 0.8 / len(groups)

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(columns) + 2), 4.5))
    for i, (group, glabel) in enumerate(groups):
        means, stds = [], []
        for s, kind, _ in columns:
            runs = collect_runs(payloads[s], kind, group)
            means.append(runs.mean() * 1e3 if runs.size else np.nan)
            stds.append(runs.std() * 1e3 if runs.size else 0.0)
        offset = (i - (len(groups) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=4, label=glabel)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n{klabel}" for s, _, klabel in columns])
    ax.set_ylabel("avg runtime (ms)")
    ax.set_title("Average matchTemplate runtime by system")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_path = os.path.join(HERE, "avg_runtime_by_system.png")
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
