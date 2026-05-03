"""
Single-build benchmark worker.

Imports a specific cv2 (controlled by --cv2-path) and runs masked
matchTemplate timings for every scenario in --scenarios-file. After
timing, optionally writes annotated images to --annotate-dir using the
same cv2 — these are NOT included in the timing measurements.

Emits a JSON payload describing the run to --out.
"""

import argparse
import json
import os
import statistics
import sys
import time


METHODS = ["TM_CCORR_NORMED", "TM_SQDIFF_NORMED"]
WARMUP = 1
RUNS = 3


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def import_cv2(cv2_path):
    if cv2_path:
        sys.path.insert(0, cv2_path)
    import cv2
    return cv2


def synthetic_inputs(cv2, img_size=2048, tpl_size=128, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (img_size, img_size, 3), dtype=np.uint8)
    ty = (img_size - tpl_size) // 2
    tx = (img_size - tpl_size) // 2
    tpl = img[ty:ty + tpl_size, tx:tx + tpl_size].copy()
    mask = np.zeros((tpl_size, tpl_size), dtype=np.uint8)
    cv2.circle(mask, (tpl_size // 2, tpl_size // 2), tpl_size // 2, 255, -1)
    return img, tpl, mask


def file_inputs(cv2, image_path, template_path, mask_path):
    import numpy as np
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"could not read image: {image_path}")
    tpl = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if tpl is None:
        raise RuntimeError(f"could not read template: {template_path}")
    if tpl.shape[0] > img.shape[0] or tpl.shape[1] > img.shape[1]:
        raise RuntimeError(
            f"template {tpl.shape[:2]} larger than image {img.shape[:2]}")
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask.shape[:2] != tpl.shape[:2]:
            mask = cv2.resize(mask, (tpl.shape[1], tpl.shape[0]))
    else:
        mask = np.full(tpl.shape[:2], 255, dtype=np.uint8)
    return img, tpl, mask


def load_scenario(cv2, sc):
    if sc.get("kind", "synthetic") == "synthetic":
        return synthetic_inputs(cv2,
                                sc.get("img_size", 2048),
                                sc.get("tpl_size", 128),
                                sc.get("seed", 0))
    return file_inputs(cv2, sc["image"], sc["template"], sc.get("mask"))


def time_it(fn):
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    mean = sum(times) / len(times)
    std = statistics.pstdev(times) if len(times) > 1 else 0.0
    return {"mean": mean, "std": std, "runs": list(times)}


def bench_cpu(cv2, img, tpl, mask, method):
    prev = None
    if cv2.ocl.haveOpenCL():
        prev = cv2.ocl.useOpenCL()
        cv2.ocl.setUseOpenCL(False)
    try:
        run = lambda: cv2.matchTemplate(img, tpl, method, mask=mask)
        out = run()
        return time_it(run), out
    finally:
        if prev is not None:
            cv2.ocl.setUseOpenCL(prev)


def bench_umat(cv2, img, tpl, mask, method):
    cv2.ocl.setUseOpenCL(True)
    u_img, u_tpl, u_mask = cv2.UMat(img), cv2.UMat(tpl), cv2.UMat(mask)

    def run():
        return cv2.matchTemplate(u_img, u_tpl, method, mask=u_mask).get()

    out = run()
    return time_it(run), out


def bench_cuda(cv2, img, tpl, mask, method):
    g_img = cv2.cuda_GpuMat();  g_img.upload(img)
    g_tpl = cv2.cuda_GpuMat();  g_tpl.upload(tpl)
    g_mask = cv2.cuda_GpuMat(); g_mask.upload(mask)

    matcher = None
    try:
        matcher = cv2.cuda.createTemplateMatching(img.dtype.num, method)
    except Exception:
        pass

    def run():
        if matcher is not None:
            try:
                return matcher.match(g_img, g_tpl, mask=g_mask).download()
            except TypeError:
                pass
        return cv2.cuda.matchTemplate(g_img, g_tpl, method, mask=g_mask).download()

    out = run()
    return time_it(run), out


BENCHERS = {"cpu": bench_cpu, "umat": bench_umat, "cuda": bench_cuda}


def annotate_scenarios(cv2, scenarios, out_dir, count=3):
    """Draw best-match boxes on a few file scenarios. Not timed."""
    import numpy as np
    file_scs = [s for s in scenarios if s.get("kind") == "files"]
    if not file_scs:
        return
    step = max(1, len(file_scs) // count)
    chosen = file_scs[::step][:count]

    os.makedirs(out_dir, exist_ok=True)
    method = cv2.TM_CCORR_NORMED
    for sc in chosen:
        try:
            img, tpl, mask = load_scenario(cv2, sc)
        except Exception as e:
            log(f"[annotate] skip {sc['name']}: {e}")
            continue
        res = cv2.matchTemplate(img, tpl, method, mask=mask)
        res = np.where(np.isfinite(res), res, -np.inf)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        h, w = tpl.shape[:2]
        annotated = img.copy()
        cv2.rectangle(annotated, max_loc,
                      (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 3)
        label = f"{os.path.basename(sc['template'])}  score={max_val:.3f}"
        cv2.putText(annotated, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(annotated, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # template thumbnail in top-right
        th, tw = tpl.shape[:2]
        x0 = annotated.shape[1] - tw - 10
        annotated[10:10 + th, x0:x0 + tw] = tpl
        cv2.rectangle(annotated, (x0 - 1, 9),
                      (x0 + tw + 1, 11 + th), (255, 255, 0), 2)

        out = os.path.join(out_dir, f"annotated__{sc['name']}.png")
        cv2.imwrite(out, annotated)
        log(f"[annotate] wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cv2-path", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--backends", nargs="+", required=True,
                   choices=["cpu", "umat", "cuda"])
    p.add_argument("--scenarios-file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--results-dir", required=True,
                   help="Directory to dump raw .npy outputs for integrity check.")
    p.add_argument("--annotate-dir", default=None,
                   help="If set, draw best-match boxes on a few file scenarios "
                        "after timing (does not affect benchmark timings).")
    args = p.parse_args()

    cv2 = import_cv2(args.cv2_path)

    method_map = {
        "TM_SQDIFF":        cv2.TM_SQDIFF,
        "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
        "TM_CCORR":         cv2.TM_CCORR,
        "TM_CCORR_NORMED":  cv2.TM_CCORR_NORMED,
        "TM_CCOEFF":        cv2.TM_CCOEFF,
        "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
    }

    have_opencl = bool(cv2.ocl.haveOpenCL())
    cuda_count = (cv2.cuda.getCudaEnabledDeviceCount()
                  if hasattr(cv2, "cuda") else 0)

    log(f"[{args.label}] cv2 from {cv2.__file__}")
    log(f"[{args.label}] version={cv2.__version__} OpenCL={have_opencl} "
        f"CUDA devices={cuda_count}")

    with open(args.scenarios_file) as fh:
        scenarios = json.load(fh)

    backends = []
    for b in args.backends:
        if b == "umat" and not have_opencl:
            log(f"[{args.label}] skipping UMat (no OpenCL)")
            continue
        if b == "cuda" and cuda_count == 0:
            log(f"[{args.label}] skipping CUDA (no device)")
            continue
        backends.append(b)

    import numpy as np
    scenarios_out = []
    for sc in scenarios:
        sc_name = sc["name"]
        try:
            img, tpl, mask = load_scenario(cv2, sc)
        except Exception as e:
            log(f"[{args.label}] scenario {sc_name} load FAILED: {e}")
            scenarios_out.append({"name": sc_name, "spec": sc,
                                  "error": str(e), "results": {}})
            continue
        log(f"[{args.label}] scenario {sc_name}: img={img.shape} tpl={tpl.shape}")

        results = {}
        for name in METHODS:
            method = method_map[name]
            results[name] = {}
            for b in backends:
                try:
                    t, out = BENCHERS[b](cv2, img, tpl, mask, method)
                    results[name][b] = t
                    fname = f"{args.label}__{sc_name}__{name}__{b}.npy"
                    np.save(os.path.join(args.results_dir, fname),
                            np.asarray(out))
                    log(f"[{args.label}] {sc_name:<28} {name:<18} {b:<5} "
                        f"{t['mean'] * 1e3:8.2f} ± {t['std'] * 1e3:6.2f} ms")
                except Exception as e:
                    results[name][b] = None
                    log(f"[{args.label}] {sc_name} {name} {b} ERROR: {e}")
        scenarios_out.append({
            "name": sc_name, "spec": sc,
            "img_shape": list(img.shape),
            "tpl_shape": list(tpl.shape),
            "results": results,
        })

    if args.annotate_dir:
        log(f"[{args.label}] annotating scenarios -> {args.annotate_dir}")
        annotate_scenarios(cv2, scenarios, args.annotate_dir)

    payload = {
        "label": args.label,
        "cv2_file": cv2.__file__,
        "cv2_version": cv2.__version__,
        "have_opencl": have_opencl,
        "cuda_devices": cuda_count,
        "warmup": WARMUP,
        "runs": RUNS,
        "backends": backends,
        "methods": METHODS,
        "scenarios": scenarios_out,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh)


if __name__ == "__main__":
    main()
