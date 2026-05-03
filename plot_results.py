"""
Plot benchmark results from bench_results.json.

One PNG per scenario in bench_plots/, grouped bar chart with std error bars.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "bench_results.json")
PLOTS_DIR = os.path.join(HERE, "bench_plots")

SERIES = [
    ("orig CPU",  "original", "cpu"),
    ("new CPU",   "new",      "cpu"),
    ("orig UMat", "original", "umat"),
    ("new UMat",  "new",      "umat"),
    ("new CUDA",  "new",      "cuda"),
]


def index_scenarios(payload):
    return {sc["name"]: sc for sc in payload["scenarios"]}


def plot_scenario(sc_name, methods, by_label, out_path):
    n_methods = len(methods)
    n_series = len(SERIES)
    width = 0.8 / n_series
    x = np.arange(n_methods)

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * n_methods + 4), 5))
    for i, (series_name, build, backend) in enumerate(SERIES):
        res = (by_label.get(build) or {}).get("results", {})
        means, stds = [], []
        for m in methods:
            entry = res.get(m, {}).get(backend)
            if entry is None:
                means.append(np.nan); stds.append(0.0)
            else:
                means.append(entry["mean"] * 1e3)
                stds.append(entry["std"] * 1e3)
        offset = (i - (n_series - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=stds,
                      label=series_name, capsize=3)
        for rect, val in zip(bars, means):
            if not np.isnan(val):
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height(), f"{val:.1f}",
                        ha="center", va="bottom", fontsize=7)

    img_shape = (by_label.get("original") or {}).get("img_shape") \
        or (by_label.get("new") or {}).get("img_shape")
    tpl_shape = (by_label.get("original") or {}).get("tpl_shape") \
        or (by_label.get("new") or {}).get("tpl_shape")
    ax.set_title(f"{sc_name}\nimg={img_shape}  tpl={tpl_shape}")
    ax.set_ylabel("ms per call (mean ± std)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    with open(DATA_FILE) as fh:
        data = json.load(fh)

    methods = data["methods"]
    orig_idx = index_scenarios(data["original"])
    new_idx = index_scenarios(data["new"])

    os.makedirs(PLOTS_DIR, exist_ok=True)
    names = list(orig_idx) + [n for n in new_idx if n not in orig_idx]

    for sc_name in names:
        out_path = os.path.join(
            PLOTS_DIR, sc_name.replace(os.sep, "_") + ".png")
        plot_scenario(sc_name, methods,
                      {"original": orig_idx.get(sc_name, {}),
                       "new":      new_idx.get(sc_name, {})},
                      out_path)


if __name__ == "__main__":
    sys.exit(main())
