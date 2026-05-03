"""
Plot benchmark results saved by masked_template_match.py.

Reads bench_results.json (default) and produces one grouped bar chart per
scenario, with bars for each (build, backend) series and error bars from
the per-run std. Writes one PNG per scenario to --out-dir.
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


SERIES = [
    ("orig CPU",  "original", "cpu"),
    ("new CPU",   "new",      "cpu"),
    ("orig UMat", "original", "umat"),
    ("new UMat",  "new",      "umat"),
    ("new CUDA",  "new",      "cuda"),
]


def index_scenarios(payload):
    return {sc["name"]: sc for sc in payload["scenarios"]}


def plot_scenario(sc_name, methods, orig_sc, new_sc, out_path):
    by_label = {"original": orig_sc, "new": new_sc}
    n_methods = len(methods)
    n_series = len(SERIES)
    width = 0.8 / n_series
    x = np.arange(n_methods)

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * n_methods + 4), 5))
    for i, (series_name, build, backend) in enumerate(SERIES):
        sc = by_label.get(build) or {}
        res = sc.get("results", {})
        means, stds = [], []
        for m in methods:
            entry = res.get(m, {}).get(backend)
            if entry is None:
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(entry["mean"] * 1e3)
                stds.append(entry["std"] * 1e3)
        offset = (i - (n_series - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, label=series_name,
                      capsize=3)
        for rect, val in zip(bars, means):
            if not np.isnan(val):
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height(), f"{val:.1f}",
                        ha="center", va="bottom", fontsize=7)

    img_shape = (orig_sc.get("img_shape") or new_sc.get("img_shape"))
    tpl_shape = (orig_sc.get("tpl_shape") or new_sc.get("tpl_shape"))
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
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(here, "bench_results.json"))
    p.add_argument("--out-dir", default=os.path.join(here, "bench_plots"))
    args = p.parse_args()

    with open(args.data) as fh:
        data = json.load(fh)

    methods = data["methods"]
    orig_idx = index_scenarios(data["original"])
    new_idx = index_scenarios(data["new"])

    os.makedirs(args.out_dir, exist_ok=True)
    names = list(orig_idx.keys())
    for n in new_idx:
        if n not in orig_idx:
            names.append(n)

    for sc_name in names:
        out_path = os.path.join(
            args.out_dir,
            sc_name.replace(os.sep, "_") + ".png")
        plot_scenario(sc_name, methods,
                      orig_idx.get(sc_name, {}),
                      new_idx.get(sc_name, {}),
                      out_path)


if __name__ == "__main__":
    sys.exit(main())
