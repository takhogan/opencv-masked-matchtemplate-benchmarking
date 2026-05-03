"""
Single-build benchmark worker.

Loads a specific cv2 build (controlled by --cv2-path) and runs masked
matchTemplate benchmarks on a list of scenarios. Each scenario is either:
  {"name": "...", "kind": "synthetic", "img_size": N, "tpl_size": M, "seed": K}
  {"name": "...", "kind": "files", "image": "...", "template": "...",
                   "mask": "..."|null}

Scenarios come from --scenarios-file (a JSON list). If absent, a single
synthetic scenario is built from --img-size / --tpl-size.

Emits a single JSON object on stdout. Human-readable progress goes to stderr.
"""

import argparse
import json
import os
import sys
import time


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def import_cv2(cv2_path):
    if cv2_path:
        sys.path.insert(0, cv2_path)
    import cv2
    return cv2


def maybe_enable_templmatch_trace():
    if os.environ.get("OCV_BENCH_TEMPLMATCH"):
        log("[trace] OCV_BENCH_TEMPLMATCH=1 — per-phase timings will follow")


def synthetic_inputs(cv2, img_size, tpl_size, seed=0):
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
    kind = sc.get("kind", "synthetic")
    if kind == "synthetic":
        return synthetic_inputs(cv2,
                                sc.get("img_size", 2048),
                                sc.get("tpl_size", 128),
                                sc.get("seed", 0))
    if kind == "files":
        return file_inputs(cv2, sc["image"], sc["template"], sc.get("mask"))
    raise ValueError(f"unknown scenario kind: {kind}")


def time_it(fn, warmup, runs):
    import statistics
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    mean = sum(times) / len(times)
    std = statistics.pstdev(times) if len(times) > 1 else 0.0
    return {"mean": mean, "std": std, "runs": list(times)}


def bench_cpu(cv2, img, tpl, mask, method, warmup, runs):
    prev = None
    if cv2.ocl.haveOpenCL():
        prev = cv2.ocl.useOpenCL()
        cv2.ocl.setUseOpenCL(False)
    try:
        run = lambda: cv2.matchTemplate(img, tpl, method, mask=mask)
        out = run()
        return time_it(run, warmup, runs), out
    finally:
        if prev is not None:
            cv2.ocl.setUseOpenCL(prev)


def bench_umat(cv2, img, tpl, mask, method, warmup, runs):
    cv2.ocl.setUseOpenCL(True)
    u_img = cv2.UMat(img)
    u_tpl = cv2.UMat(tpl)
    u_mask = cv2.UMat(mask)

    def run():
        r = cv2.matchTemplate(u_img, u_tpl, method, mask=u_mask)
        return r.get()

    out = run()
    return time_it(run, warmup, runs), out


def bench_cuda(cv2, img, tpl, mask, method, warmup, runs):
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
                r = matcher.match(g_img, g_tpl, mask=g_mask)
                return r.download()
            except TypeError:
                pass
        r = cv2.cuda.matchTemplate(g_img, g_tpl, method, mask=g_mask)
        return r.download()

    out = run()
    return time_it(run, warmup, runs), out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cv2-path", default=None)
    p.add_argument("--label", required=True)
    p.add_argument("--backends", nargs="+",
                   default=["cpu", "umat", "cuda"],
                   choices=["cpu", "umat", "cuda"])
    p.add_argument("--methods", nargs="+",
                   default=["TM_CCORR_NORMED", "TM_SQDIFF_NORMED"])
    p.add_argument("--scenarios-file", default=None,
                   help="JSON list of scenarios. If absent, a single synthetic "
                        "scenario is built from --img-size/--tpl-size.")
    p.add_argument("--img-size", type=int, default=2048)
    p.add_argument("--tpl-size", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--trace-templmatch", action="store_true",
                   help="Set OCV_BENCH_TEMPLMATCH=1 and run a single warmup+1-run "
                        "trace per (scenario,method,backend) to find bottlenecks.")
    p.add_argument("--out", default=None,
                   help="If set, write JSON results here instead of stdout. "
                        "Useful when verbose OpenCL logs spam stdout.")
    p.add_argument("--results-dir", default=None,
                   help="If set, save raw output arrays as .npy files here for "
                        "an integrity check by the orchestrator.")
    args = p.parse_args()

    if args.trace_templmatch:
        os.environ["OCV_BENCH_TEMPLMATCH"] = "1"
        args.warmup = 1
        args.runs = 1

    cv2 = import_cv2(args.cv2_path)
    maybe_enable_templmatch_trace()

    methods = {
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
    have_cuda = cuda_count > 0

    log(f"[{args.label}] cv2 from {cv2.__file__}")
    log(f"[{args.label}] version={cv2.__version__} OpenCL={have_opencl} "
        f"CUDA devices={cuda_count}")

    if args.scenarios_file:
        with open(args.scenarios_file) as fh:
            scenarios = json.load(fh)
    else:
        scenarios = [{
            "name": f"synthetic_{args.img_size}x{args.tpl_size}",
            "kind": "synthetic",
            "img_size": args.img_size,
            "tpl_size": args.tpl_size,
        }]

    backends = []
    for b in args.backends:
        if b == "umat" and not have_opencl:
            log(f"[{args.label}] skipping UMat (no OpenCL)")
            continue
        if b == "cuda" and not have_cuda:
            log(f"[{args.label}] skipping CUDA (no device)")
            continue
        backends.append(b)

    benchers = {"cpu": bench_cpu, "umat": bench_umat, "cuda": bench_cuda}

    scenarios_out = []
    for sc in scenarios:
        sc_name = sc["name"]
        try:
            img, tpl, mask = load_scenario(cv2, sc)
        except Exception as e:
            log(f"[{args.label}] scenario {sc_name} load FAILED: {e}")
            scenarios_out.append({
                "name": sc_name, "spec": sc, "error": str(e), "results": {},
            })
            continue
        log(f"[{args.label}] scenario {sc_name}: img={img.shape} tpl={tpl.shape}")

        results = {}
        for name in args.methods:
            if name not in methods:
                log(f"[{args.label}] unknown method {name}, skipping")
                continue
            method = methods[name]
            results[name] = {}
            for b in backends:
                try:
                    t, out = benchers[b](cv2, img, tpl, mask, method,
                                         args.warmup, args.runs)
                    results[name][b] = t
                    if args.results_dir is not None and out is not None:
                        import numpy as np
                        fname = f"{args.label}__{sc_name}__{name}__{b}.npy"
                        np.save(os.path.join(args.results_dir, fname),
                                np.asarray(out))
                    log(f"[{args.label}] {sc_name:<28} {name:<18} {b:<5} "
                        f"{t['mean'] * 1e3:8.2f} ± {t['std'] * 1e3:6.2f} ms")
                except Exception as e:
                    results[name][b] = None
                    log(f"[{args.label}] {sc_name} {name} {b} ERROR: {e}")
        scenarios_out.append({
            "name": sc_name,
            "spec": sc,
            "img_shape": list(img.shape),
            "tpl_shape": list(tpl.shape),
            "results": results,
        })

    payload = {
        "label": args.label,
        "cv2_file": cv2.__file__,
        "cv2_version": cv2.__version__,
        "have_opencl": have_opencl,
        "cuda_devices": cuda_count,
        "warmup": args.warmup,
        "runs": args.runs,
        "backends": backends,
        "scenarios": scenarios_out,
    }
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload, fh)
    else:
        json.dump(payload, sys.stdout)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
