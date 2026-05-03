#!/usr/bin/env python3
"""
Drop near-duplicate stills: frames that match the previous *kept* frame above a
similarity threshold (sequential duplicate detection).

Compared on small grayscale thumbnails for speed — good for bursts of visually
flat video. Lower --min-similarity if subtle changes should still count as new.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.join(HERE, "data", "images")
DEFAULT_OUT = os.path.join(HERE, "data", "images_deduped")

_FRAME_RE = re.compile(
    r"^(?P<base>.+)_frame_(?P<idx>\d+)\.(?P<ext>png|jpe?g)$", re.IGNORECASE
)


def group_image_paths(image_dir: str) -> dict[str, list[str]]:
    """Group paths by `{base}` for names like `maps_test_frame_000042.png`."""
    by_base: dict[str, list[str]] = {}
    for name in sorted(os.listdir(image_dir)):
        m = _FRAME_RE.match(name)
        if not m:
            continue
        path = os.path.join(image_dir, name)
        if not os.path.isfile(path):
            continue
        base = m.group("base")
        by_base.setdefault(base, []).append(path)

    for paths in by_base.values():

        def _idx(pp: str) -> int:
            mm = _FRAME_RE.match(os.path.basename(pp))
            assert mm is not None
            return int(mm.group("idx"))

        paths.sort(key=_idx)
    return by_base


def load_compare_gray(path: str, max_side: int) -> object:
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(
            img,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float64")
    return gray / 255.0


def similarity(a: object, b: object) -> float:
    import numpy as np

    # Pad or crop smallest to larger if codec sizes drift (usually identical).
    ah, aw = a.shape[:2]
    bh, bw = b.shape[:2]
    if (ah, aw) != (bh, bw):
        h, w = min(ah, bh), min(aw, bw)
        a = a[:h, :w]
        b = b[:h, :w]
    return float(1.0 - np.mean(np.abs(a - b)))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Remove consecutive near-duplicate frame images.",
    )
    p.add_argument(
        "--in",
        dest="in_dir",
        default=DEFAULT_IN,
        help=f"Directory of PNG/JPEG frames (default: {DEFAULT_IN})",
    )
    p.add_argument(
        "--out",
        dest="out_dir",
        default=DEFAULT_OUT,
        help=f"Deduplicated frames written here (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--min-similarity",
        type=float,
        default=0.99,
        metavar="X",
        help=(
            "If similarity to previous kept frame >= X, skip (duplicate). "
            "Default 0.99 (~1%% different pixels on average)."
        ),
    )
    p.add_argument(
        "--compare-max-side",
        type=int,
        default=160,
        metavar="PX",
        help="Downscale longest side to this before comparing (speed). Default 160.",
    )
    p.add_argument(
        "--no-renumber",
        action="store_true",
        help="Keep original filenames; otherwise rename to contiguous _frame_%06d.",
    )
    args = p.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isdir(in_dir):
        print(f"Not a directory: {in_dir}", file=sys.stderr)
        raise SystemExit(1)

    ext_re = _FRAME_RE

    grouped = group_image_paths(in_dir)
    if not grouped:
        print(
            f"No *_frame_*.png|jpg files matched in {in_dir}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    os.makedirs(out_dir, exist_ok=True)
    thresh = args.min_similarity
    mx = args.compare_max_side
    renumber = not args.no_renumber

    total_in = total_out = 0
    for base, paths in sorted(grouped.items()):
        kept_ref = None  # ndarray
        seq_out = 0
        skipped = 0
        for path in paths:
            total_in += 1
            gray = load_compare_gray(path, mx)
            if kept_ref is not None:
                sim = similarity(gray, kept_ref)
                if sim >= thresh:
                    skipped += 1
                    continue
            kept_ref = gray

            bm = os.path.basename(path)
            m = ext_re.match(bm)
            ext = "png"
            if m is not None:
                ext = m.group("ext").lower()
            if ext == "jpeg":
                ext = "jpg"
            if renumber:
                dst = os.path.join(out_dir, f"{base}_frame_{seq_out:06d}.{ext}")
            else:
                dst = os.path.join(out_dir, bm)
            shutil.copy2(path, dst)
            seq_out += 1
            total_out += 1
        print(
            f"{base}: kept {seq_out}/{len(paths)} "
            f"({skipped} skipped, min_similarity>={thresh})"
        )

    print(f"Done. {total_in} frames in -> {total_out} frames out -> {out_dir}")


if __name__ == "__main__":
    main()
