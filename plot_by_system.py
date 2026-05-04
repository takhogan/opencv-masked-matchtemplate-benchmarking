"""Bar plots of average runtime by system, split by synthetic vs real images."""

import glob
import json
import os

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

NAVY = "#21295C"
DEEP = "#065A82"
TEAL = "#1C7293"
ICE = "#E8EEF4"
ACCENT = "#F2A65A"
TEXT = "#1F2937"
MUTED = "#64748B"

_available = {f.name for f in font_manager.fontManager.ttflist}
HEADER_FONT = "Georgia" if "Georgia" in _available else "DejaVu Serif"
BODY_FONT = "Calibri" if "Calibri" in _available else "DejaVu Sans"


# (system label, results file stem, gpu backend used for that system)
SYSTEMS = [
    ("mac",     "mac",     "umat"),
    ("windows", "windows", "umat"),
    ("linux",   "linux",   "umat"),
    ("cuda",    "linux",   "cuda"),
]


def collect_runs(payload, kind, backend_filter, build_name="new"):
    runs = []
    build = payload.get(build_name)
    if not isinstance(build, dict):
        return np.array([])
    for sc in build.get("scenarios", []):
        if sc.get("spec", {}).get("kind") != kind:
            continue
        for method_res in sc.get("results", {}).values():
            entry = method_res.get(backend_filter)
            if entry:
                runs.extend(entry.get("runs", []))
    return np.array(runs, dtype=float)


def main():
    payloads = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_bench_results.json"))):
        name = os.path.basename(path).split("_bench_results.json")[0]
        with open(path) as fh:
            payloads[name] = json.load(fh)

    systems = [(label, payloads[stem], gpu) for label, stem, gpu in SYSTEMS
               if stem in payloads]

    kinds = [("synthetic", "synthetic"), ("files", "real")]
    groups = [("cpu", "CPU", TEAL), ("gpu", "GPU", ACCENT)]
    columns = [(label, payload, gpu, kind, klabel)
               for label, payload, gpu in systems
               for kind, klabel in kinds]
    x = np.arange(len(columns))
    width = 0.8 / len(groups)

    fig, ax = plt.subplots(figsize=(max(7.5, 1.5 * len(columns) + 2), 5),
                           facecolor="white")
    ax.set_facecolor(ICE)

    for i, (group, glabel, color) in enumerate(groups):
        means, stds = [], []
        for label, payload, gpu, kind, _ in columns:
            backend = "cpu" if group == "cpu" else gpu
            runs = collect_runs(payload, kind, backend)
            means.append(runs.mean() * 1e3 if runs.size else np.nan)
            stds.append(runs.std() * 1e3 if runs.size else 0.0)
        offset = (i - (len(groups) - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4,
                      label=glabel, color=color, edgecolor=NAVY, linewidth=0.8,
                      error_kw=dict(ecolor=NAVY, elinewidth=1.1))
        for rect, val in zip(bars, means):
            if not np.isnan(val):
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height(), f"{val:.1f}",
                        ha="center", va="bottom", fontsize=8,
                        color=TEXT, fontfamily=BODY_FONT)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{label}\n{klabel}"
                        for label, _, _, _, klabel in columns],
                       fontfamily=BODY_FONT, color=TEXT, fontsize=10)
    ax.set_ylabel("avg runtime (ms)", fontfamily=BODY_FONT,
                  color=TEXT, fontsize=11)
    ax.set_title("Average matchTemplate runtime by system",
                 fontfamily=HEADER_FONT, color=NAVY, fontsize=15,
                 fontweight="bold", pad=12)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", linestyle="-", color="white", linewidth=1.2)
    ax.set_axisbelow(True)

    legend = ax.legend(frameon=False, loc="upper left",
                       prop={"family": BODY_FONT, "size": 10})
    for text in legend.get_texts():
        text.set_color(DEEP)

    fig.tight_layout()
    out_path = os.path.join(HERE, "avg_runtime_by_system.png")
    fig.savefig(out_path, dpi=150, facecolor="white")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
