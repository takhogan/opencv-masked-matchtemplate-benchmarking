"""
masked matchTemplate benchmark.

Spawns the worker (_bench_worker.py) once per cv2 build:
  * --original-cv2   path to the cv2 module built from original_opencv
  * --new-cv2        path to the cv2 module built from new_opencv

Each child process imports its own cv2, runs CPU / UMat / CUDA timings on
every scenario, and emits JSON. The orchestrator aggregates and prints a
side-by-side comparison per scenario:

    UMat  : original_opencv   vs   new_opencv
    CUDA  : new_opencv only

Scenarios:
  * always: one synthetic scenario (--img-size / --tpl-size)
  * if data/imgs/ and data/templates/ exist and contain images,
    one scenario for every (image, template) combination. An optional
    same-named file in data/masks/ is used as the mask; otherwise a
    full-on mask is generated.

Disable real-image scenarios with --no-pairs.
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
IMG_DIR = os.path.join(DATA_DIR, "imgs")
TPL_DIR = os.path.join(DATA_DIR, "templates")
MASK_DIR = os.path.join(DATA_DIR, "masks")

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def list_images(d):
    if not os.path.isdir(d):
        return []
    return sorted(
        os.path.join(d, f)
        for f in os.listdir(d)
        if f.lower().endswith(IMAGE_EXTS) and not f.startswith(".")
    )


def find_mask_for(template_path):
    if not os.path.isdir(MASK_DIR):
        return None
    base = os.path.splitext(os.path.basename(template_path))[0]
    for ext in IMAGE_EXTS:
        cand = os.path.join(MASK_DIR, base + ext)
        if os.path.exists(cand):
            return cand
    return None


def build_scenarios(img_size, tpl_size, include_pairs):
    scenarios = [{
        "name": f"synthetic_{img_size}x{tpl_size}",
        "kind": "synthetic",
        "img_size": img_size,
        "tpl_size": tpl_size,
    }]
    if not include_pairs:
        return scenarios

    imgs = list_images(IMG_DIR)
    tpls = list_images(TPL_DIR)
    if not imgs or not tpls:
        return scenarios

    for img_path, tpl_path in itertools.product(imgs, tpls):
        ib = os.path.splitext(os.path.basename(img_path))[0]
        tb = os.path.splitext(os.path.basename(tpl_path))[0]
        scenarios.append({
            "name": f"{ib}__x__{tb}",
            "kind": "files",
            "image": os.path.abspath(img_path),
            "template": os.path.abspath(tpl_path),
            "mask": find_mask_for(tpl_path),
        })
    return scenarios


def run_worker(python_exe, cv2_path, label, backends, methods,
               scenarios_file, warmup, runs, results_dir=None):
    out_fd, out_path = tempfile.mkstemp(prefix=f"bench_{label}_", suffix=".json")
    os.close(out_fd)

    cmd = [python_exe, WORKER,
           "--label", label,
           "--backends", *backends,
           "--methods", *methods,
           "--scenarios-file", scenarios_file,
           "--warmup", str(warmup),
           "--runs", str(runs),
           "--out", out_path]
    if results_dir:
        cmd += ["--results-dir", results_dir]
    env = os.environ.copy()
    if cv2_path:
        cmd += ["--cv2-path", cv2_path]
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


def _std(s):
    return s["std"] if s else None


def fmt_ms(s):
    if s is None:
        return f"{'-':>15}"
    return f"{s['mean'] * 1e3:7.2f}±{s['std'] * 1e3:5.2f}"


def fmt_speedup(base, t):
    if base is None or t is None or t == 0:
        return f"{'-':>7}"
    return f"{base / t:6.2f}x"


def index_scenarios(payload):
    return {sc["name"]: sc for sc in payload["scenarios"]}


def print_comparison(orig, new, methods):
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
    names = list(o_idx.keys())
    for n in n_idx:
        if n not in o_idx:
            names.append(n)

    header = (f"{'method':<20}"
              f"{'orig CPU':>16}{'new CPU':>16}"
              f"{'orig UMat':>16}{'new UMat':>16}{'UMat new/orig':>15}"
              f"{'new CUDA':>16}{'CUDA vs UMat':>14}")

    for sc_name in names:
        o_sc = o_idx.get(sc_name, {})
        n_sc = n_idx.get(sc_name, {})
        o_shape = o_sc.get("img_shape") or n_sc.get("img_shape")
        t_shape = o_sc.get("tpl_shape") or n_sc.get("tpl_shape")
        print()
        print(f"--- scenario: {sc_name}  img={o_shape}  tpl={t_shape}")
        if o_sc.get("error"):
            print(f"    original load error: {o_sc['error']}")
        if n_sc.get("error"):
            print(f"    new load error: {n_sc['error']}")
        print(header)
        print("-" * len(header))

        o_res = o_sc.get("results", {})
        n_res = n_sc.get("results", {})
        for m in methods:
            o = o_res.get(m, {})
            n = n_res.get(m, {})
            o_cpu, n_cpu = o.get("cpu"), n.get("cpu")
            o_umat, n_umat = o.get("umat"), n.get("umat")
            n_cuda = n.get("cuda")
            o_umat_m, n_umat_m, n_cuda_m = _mean(o_umat), _mean(n_umat), _mean(n_cuda)
            umat_ratio = (f"{o_umat_m / n_umat_m:6.2f}x"
                          if (o_umat_m and n_umat_m) else "-")
            cuda_ratio = (f"{n_umat_m / n_cuda_m:6.2f}x"
                          if (n_umat_m and n_cuda_m) else "-")
            print(f"{m:<20}"
                  f"{fmt_ms(o_cpu):>16}{fmt_ms(n_cpu):>16}"
                  f"{fmt_ms(o_umat):>16}{fmt_ms(n_umat):>16}{umat_ratio:>15}"
                  f"{fmt_ms(n_cuda):>16}{cuda_ratio:>14}")


def integrity_check(orig, new, methods, results_dir,
                    rtol=1e-4, atol=1e-5):
    """Compare new vs original output arrays for each scenario/method/backend.

    Runs after timing so it doesn't affect runtime measurements. Compares
    every (scenario, method, backend) where both workers produced an output.
    """
    import numpy as np

    print()
    print("=" * 96)
    print(f"Integrity check (new vs original, rtol={rtol}, atol={atol})")
    print("=" * 96)

    o_idx = index_scenarios(orig)
    n_idx = index_scenarios(new)
    common = [s for s in o_idx if s in n_idx]

    total = passed = failed = missing = 0
    for sc_name in common:
        o_res = o_idx[sc_name].get("results", {})
        n_res = n_idx[sc_name].get("results", {})
        for m in methods:
            o_b = o_res.get(m, {})
            n_b = n_res.get(m, {})
            backends = set(o_b) & set(n_b)
            for b in sorted(backends):
                if _mean(o_b.get(b)) is None or _mean(n_b.get(b)) is None:
                    continue
                o_path = os.path.join(
                    results_dir, f"original__{sc_name}__{m}__{b}.npy")
                n_path = os.path.join(
                    results_dir, f"new__{sc_name}__{m}__{b}.npy")
                if not (os.path.exists(o_path) and os.path.exists(n_path)):
                    missing += 1
                    print(f"  MISS  {sc_name} {m} {b}: array file not found")
                    continue
                total += 1
                a = np.load(o_path)
                c = np.load(n_path)
                if a.shape != c.shape:
                    failed += 1
                    print(f"  FAIL  {sc_name} {m} {b}: shape {a.shape} vs {c.shape}")
                    continue
                if np.allclose(a, c, rtol=rtol, atol=atol, equal_nan=True):
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
    p.add_argument("--original-cv2", default=None)
    p.add_argument("--new-cv2", default=None)
    p.add_argument("--original-python", default=sys.executable)
    p.add_argument("--new-python", default=sys.executable)
    p.add_argument("--methods", nargs="+",
                   default=["TM_CCORR_NORMED", "TM_SQDIFF_NORMED"])
    p.add_argument("--img-size", type=int, default=2048,
                   help="Synthetic scenario image side length.")
    p.add_argument("--tpl-size", type=int, default=128,
                   help="Synthetic scenario template side length.")
    p.add_argument("--no-pairs", action="store_true",
                   help="Skip the data/imgs x data/templates pairs.")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--save-data", default=os.path.join(HERE, "bench_results.json"),
                   help="Path to write aggregated benchmark data (mean/std per "
                        "scenario/method/backend) for plot_results.py.")
    args = p.parse_args()

    scenarios = build_scenarios(args.img_size, args.tpl_size,
                                include_pairs=not args.no_pairs)
    pair_count = len(scenarios) - 1
    print(f"==> scenarios: 1 synthetic + {pair_count} image/template pair(s)",
          file=sys.stderr)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(scenarios, fh)
        scen_path = fh.name

    results_dir = tempfile.mkdtemp(prefix="bench_results_")

    try:
        orig = run_worker(args.original_python, args.original_cv2,
                          label="original",
                          backends=["cpu", "umat"],
                          methods=args.methods,
                          scenarios_file=scen_path,
                          warmup=args.warmup, runs=args.runs,
                          results_dir=results_dir)
        new = run_worker(args.new_python, args.new_cv2,
                         label="new",
                         backends=["cpu", "umat", "cuda"],
                         methods=args.methods,
                         scenarios_file=scen_path,
                         warmup=args.warmup, runs=args.runs,
                         results_dir=results_dir)
    finally:
        try:
            os.unlink(scen_path)
        except OSError:
            pass

    print_comparison(orig, new, args.methods)
    integrity_check(orig, new, args.methods, results_dir)
    shutil.rmtree(results_dir, ignore_errors=True)

    if args.save_data:
        with open(args.save_data, "w") as fh:
            json.dump({"original": orig, "new": new,
                       "methods": args.methods}, fh, indent=2)
        print(f"\nbenchmark data saved to {args.save_data}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
