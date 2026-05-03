#!/usr/bin/env bash
# Recreate the workspace on a fresh machine.
#
# Layout produced (relative to the parent of this script's directory):
#   <root>/
#     opencv-masked-matchtemplate-benchmarking/   # this repo (already cloned by you)
#     new_opencv/
#       opencv/              # takhogan/opencv  @ feature/opencl-masked-template-match
#       opencv_contrib/      # takhogan/opencv_contrib @ cuda-masked-template-match
#     original_opencv/
#       opencv/              # opencv/opencv @ 4.x
#       opencv_contrib/      # opencv/opencv_contrib @ 4.x
#
# Usage:
#   ./setup.sh                # https clones (default)
#   USE_SSH=1 ./setup.sh      # use git@github.com for the takhogan forks

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

if [[ "${USE_SSH:-0}" == "1" ]]; then
  FORK_OPENCV="git@github.com:takhogan/opencv.git"
  FORK_CONTRIB="git@github.com:takhogan/opencv_contrib.git"
else
  FORK_OPENCV="https://github.com/takhogan/opencv.git"
  FORK_CONTRIB="https://github.com/takhogan/opencv_contrib.git"
fi

UPSTREAM_OPENCV="https://github.com/opencv/opencv.git"
UPSTREAM_CONTRIB="https://github.com/opencv/opencv_contrib.git"

# url, dest, branch
clone_at() {
  local url="$1" dest="$2" branch="$3"
  if [[ -d "$dest/.git" ]]; then
    echo "[skip] $dest already exists"
    return 0
  fi
  echo "[clone] $url -> $dest (branch $branch)"
  mkdir -p "$(dirname "$dest")"
  git clone --branch "$branch" "$url" "$dest"
}

clone_at "$FORK_OPENCV"     "$ROOT/new_opencv/opencv"          "feature/opencl-masked-template-match"
clone_at "$FORK_CONTRIB"    "$ROOT/new_opencv/opencv_contrib"  "cuda-masked-template-match"
clone_at "$UPSTREAM_OPENCV" "$ROOT/original_opencv/opencv"          "4.x"
clone_at "$UPSTREAM_CONTRIB" "$ROOT/original_opencv/opencv_contrib" "4.x"

echo
echo "Done. Next steps:"
echo "  1. Drop your input videos / frames into $HERE/data/"
echo "  2. Build OpenCV:   $HERE/build_opencvs.sh"
echo "  3. Run benchmark:  $HERE/run.sh"
