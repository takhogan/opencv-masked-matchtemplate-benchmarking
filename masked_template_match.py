"""
Masked matchTemplate benchmark — orchestrator.

Spawns _bench_worker.py once per cv2 build (original / new), aggregates
results, prints a side-by-side comparison, runs an integrity check, and
writes:
  - bench_results.json  (mean/std per scenario/method/backend)
  - bench_plots/annotated/*.png  (best-match boxes drawn on a few images
    by the new cv2; produced AFTER timing so it doesn't affect numbers)

Scenarios:
  - one synthetic scenario (2048x128 random image with circular mask)
  - one scenario per (image, template) under data/images/ x data/templates/
    Each template is paired with its sibling <base>-mask.<ext>.

Usage:
    python masked_template_match.py --original-cv2 PATH --new-cv2 PATH
"""

import argparse
import itertools
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile


HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "_bench_worker.py")
DATA_DIR = os.path.join(HERE, "data")
IMG_DIR = os.path.join(DATA_DIR, "images")
TPL_DIR = os.path.join(DATA_DIR, "templates")

DATA_FILE = os.path.join(HERE, "bench_results.json")
PLOTS_DIR = os.path.join(HERE, "bench_plots")
ANNOTATE_DIR = os.path.join(PLOTS_DIR, "annotated")

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
MASK_SUFFIX = "-mask"
METHODS = ["TM_CCORR_NORMED", "TM_SQDIFF_NORMED"]


def list_images(d, exclude_masks=False):
    if not os.path.isdir(d):
        return []
    out = []
    for f in sorted(os.listdir(d)):
        if f.startswith(".") or not f.lower().endswith(IMAGE_EXTS):
            continue
        if exclude_masks and os.path.splitext(f)[0].endswith(MASK_SUFFIX):
            continue
        out.append(os.path.join(d, f))
    return out


def find_mask_for(template_path):
    d = os.path.dirname(template_path)
    base, ext = os.path.splitext(os.path.basename(template_path))
    for e in (ext, *IMAGE_EXTS):
        cand = os.path.join(d, f"{base}{MASK_SUFFIX}{e}")
        if os.path.exists(cand):
            return cand
    return None


def build_scenarios():
    scenarios = [{
        "name": "synthetic_2048x128",
        "kind": "synthetic",
        "img_size": 2048,
        "tpl_size": 128,
    }]
    imgs = list_images(IMG_DIR)
    tpls = list_images(TPL_DIR, exclude_masks=True)
    for img_path, tpl_path in itertools.product(imgs, tpls):
        ib = os.path.splitext(os.path.basename(img_path))[0]
        tb = os.path.splitext(os.path.basename(tpl_path))[0]
        mask_path = find_mask_for(tpl_path)
        if mask_path is None:
            print(f"WARN: no mask for {tpl_path}; using full-on mask",
                  file=sys.stderr)
        scenarios.append({
            "name": f"{ib}__x__{tb}",
            "kind": "files",
            "image": os.path.abspath(img_path),
            "template": os.path.abspath(tpl_path),
            "mask": os.path.abspath(mask_path) if mask_path else None,
        })
    return scenarios


def run_worker(cv2_path, label, backends, scenarios_file,
               results_dir, annotate_dir=None):
    out_fd, out_path = tempfile.mkstemp(prefix=f"bench_{label}_", suffix=".json")
    os.close(out_fd)

    cmd = [sys.executable, WORKER,
           "--cv2-path", cv2_path,
           "--label", label,
           "--backends", *backends,
           "--scenarios-file", scenarios_file,
           "--results-dir", results_dir,
           "--out", out_path]
    if annotate_dir:
        cmd += ["--annotate-dir", annotate_dir]
    env = os.environ.copy()
    env["PYTHONPATH"] = cv2_path + os.pathsep + env.get("PYTHONPATH", "")

    print(f"\n>>> {label}: {' '.join(shlex.quote(c) for c in cmd)}",
          file=sys.stderr)
    try:
        proc = subprocess.run(cmd, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"worker '{label}' failed (rc={proc.returncode})")
        with open(out_path) as fh:
            return json.load(fh)
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def _mean(s):
    return s["mean"] if s else None


def fmt_ms(s):
    if s is None:
        return f"{'-':>15}"
    return f"{s['mean'] * 1e3:7.2f}±{s['std'] * 1e3:5.2f}"


def index_scenarios(payload):
    return {sc["name"]: sc for sc in payload["scenarios"]}


def print_comparison(orig, new):
    print()
    print("=" * 96)
    print("Apples-to-apples masked matchTemplate (ms per call)")
    print("=" * 96)
    print(f"original cv2: {orig['cv2_file']}  v{orig['cv2_version']}  "
          f"OpenCL={orig['have_opencl']}  CUDA={orig['cuda_devices']}")
    print(f"new      cv2: {new['cv2_file']}  v{new['cv2_version']}  "
          f"OpenCL={new['have_opencl']}  CUDA={new['cuda_devices']}")
    print(f"runs={orig['runs']} (warmup={orig['warmup']})")

    o_idx = index_scenarios(orig)
    n_idx = index_scenarios(new)
    names = list(o_idx) + [n for n in n_idx if n not in o_idx]

    header = (f"{'method':<20}"
              f"{'orig CPU':>16}{'new CPU':>16}"
              f"{'orig UMat':>16}{'new UMat':>16}{'UMat new/orig':>15}"
              f"{'new CUDA':>16}{'CUDA vs UMat':>14}")

    for sc_name in names:
        o_sc = o_idx.get(sc_name, {})
        n_sc = n_idx.get(sc_name, {})
        print()
        print(f"--- scenario: {sc_name}  "
              f"img={o_sc.get('img_shape') or n_sc.get('img_shape')}  "
              f"tpl={o_sc.get('tpl_shape') or n_sc.get('tpl_shape')}")
        for sc, tag in ((o_sc, "original"), (n_sc, "new")):
            if sc.get("error"):
                print(f"    {tag} load error: {sc['error']}")
        print(header)
        print("-" * len(header))

        o_res = o_sc.get("results", {})
        n_res = n_sc.get("results", {})
        for m in METHODS:
            o, n = o_res.get(m, {}), n_res.get(m, {})
            o_umat_m, n_umat_m = _mean(o.get("umat")), _mean(n.get("umat"))
            n_cuda_m = _mean(n.get("cuda"))
            umat_ratio = (f"{o_umat_m / n_umat_m:6.2f}x"
                          if (o_umat_m and n_umat_m) else "-")
            cuda_ratio = (f"{n_umat_m / n_cuda_m:6.2f}x"
                          if (n_umat_m and n_cuda_m) else "-")
            print(f"{m:<20}"
                  f"{fmt_ms(o.get('cpu')):>16}{fmt_ms(n.get('cpu')):>16}"
                  f"{fmt_ms(o.get('umat')):>16}{fmt_ms(n.get('umat')):>16}"
                  f"{umat_ratio:>15}"
                  f"{fmt_ms(n.get('cuda')):>16}{cuda_ratio:>14}")


def integrity_check(orig, new, results_dir, rtol=1e-4, atol=1e-5):
    """Compare new vs original output arrays for each scenario/method/backend."""
    import numpy as np

    print()
    print("=" * 96)
    print(f"Integrity check (new vs original, rtol={rtol}, atol={atol})")
    print("=" * 96)

    o_idx = index_scenarios(orig)
    n_idx = index_scenarios(new)

    total = passed = failed = missing = 0
    for sc_name in (s for s in o_idx if s in n_idx):
        o_res = o_idx[sc_name].get("results", {})
        n_res = n_idx[sc_name].get("results", {})
        for m in METHODS:
            o_b = o_res.get(m, {})
            n_b = n_res.get(m, {})
            for b in sorted(set(o_b) & set(n_b)):
                if _mean(o_b.get(b)) is None or _mean(n_b.get(b)) is None:
                    continue
                o_path = os.path.join(results_dir, f"original__{sc_name}__{m}__{b}.npy")
                n_path = os.path.join(results_dir, f"new__{sc_name}__{m}__{b}.npy")
                if not (os.path.exists(o_path) and os.path.exists(n_path)):
                    missing += 1
                    print(f"  MISS  {sc_name} {m} {b}")
                    continue
                total += 1
                a, c = np.load(o_path), np.load(n_path)
                if a.shape != c.shape:
                    failed += 1
                    print(f"  FAIL  {sc_name} {m} {b}: shape {a.shape} vs {c.shape}")
                elif np.allclose(a, c, rtol=rtol, atol=atol, equal_nan=True):
                    passed += 1
                    print(f"  OK    {sc_name} {m} {b}")
                else:
                    diff = np.abs(a.astype(np.float64) - c.astype(np.float64))
                    failed += 1
                    print(f"  FAIL  {sc_name} {m} {b}: "
                          f"max|diff|={diff.max():.3e} "
                          f"mean|diff|={diff.mean():.3e}")

    print(f"\nintegrity: {passed}/{total} passed, {failed} failed, "
          f"{missing} missing")
    return failed == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original-cv2", required=True)
    p.add_argument("--new-cv2", required=True)
    args = p.parse_args()

    scenarios = build_scenarios()
    print(f"==> scenarios: {len(scenarios)} "
          f"(1 synthetic + {len(scenarios) - 1} image/template pair(s))",
          file=sys.stderr)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(scenarios, fh)
        scen_path = fh.name
    results_dir = tempfile.mkdtemp(prefix="bench_results_")

    try:
        orig = run_worker(args.original_cv2, "original",
                          backends=["cpu", "umat"],
                          scenarios_file=scen_path,
                          results_dir=results_dir)
        new = run_worker(args.new_cv2, "new",
                         backends=["cpu", "umat", "cuda"],
                         scenarios_file=scen_path,
                         results_dir=results_dir,
                         annotate_dir=ANNOTATE_DIR)
    finally:
        try:
            os.unlink(scen_path)
        except OSError:
            pass

    print_comparison(orig, new)
    integrity_check(orig, new, results_dir)
    shutil.rmtree(results_dir, ignore_errors=True)

    with open(DATA_FILE, "w") as fh:
        json.dump({"original": orig, "new": new, "methods": METHODS},
                  fh, indent=2)
    print(f"\nbenchmark data saved to {DATA_FILE}", file=sys.stderr)
    print(f"annotated images in   {ANNOTATE_DIR}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
