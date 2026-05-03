#!/usr/bin/env python3
"""
Extract every frame from MP4 files into PNG images.

Default input is data/input/maps_test.mp4. Default output is
data/images.

Examples:
  python extract_mp4_frames.py
  python extract_mp4_frames.py data/input/other.mp4
  python extract_mp4_frames.py /path/to/folder_of_mp4s --out data/images
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(HERE, "data", "input", "maps_test.mp4")
DEFAULT_OUT = os.path.join(HERE, "data", "images")


def collect_mp4_paths(path: str) -> list[str]:
    path = os.path.abspath(path)
    if os.path.isfile(path):
        if path.lower().endswith(".mp4"):
            return [path]
        raise SystemExit(f"Not an .mp4 file: {path}")
    if not os.path.isdir(path):
        raise SystemExit(f"Path not found: {path}")
    out: list[str] = []
    for name in sorted(os.listdir(path)):
        if name.lower().endswith(".mp4"):
            out.append(os.path.join(path, name))
    if not out:
        raise SystemExit(f"No .mp4 files in directory: {path}")
    return out


def extract_one(video_path: str, out_dir: str) -> int:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    stem = os.path.splitext(os.path.basename(video_path))[0]
    prefix = f"{stem}_frame_"
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fname = os.path.join(out_dir, f"{prefix}{n:06d}.png")
        if not cv2.imwrite(fname, frame):
            cap.release()
            raise RuntimeError(f"Failed to write: {fname}")
        n += 1
    cap.release()
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Dump MP4 frames to PNG images.")
    p.add_argument(
        "input_path",
        nargs="?",
        default=DEFAULT_INPUT,
        help=(
            "Directory containing .mp4 files, or a single .mp4 file "
            f"(default: {DEFAULT_INPUT})"
        ),
    )
    p.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    args = p.parse_args()

    try:
        videos = collect_mp4_paths(args.input_path)
    except SystemExit as e:
        print(e, file=sys.stderr)
        raise SystemExit(1) from None

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    total_frames = 0
    for vp in videos:
        count = extract_one(vp, out_dir)
        print(f"{vp}: {count} frames -> {out_dir}")
        total_frames += count
    print(f"Done. {len(videos)} video(s), {total_frames} frame(s).")


if __name__ == "__main__":
    main()
