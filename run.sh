#!/usr/bin/env bash
# Build cv2 from original_opencv and new_opencv if missing, then run the
# masked-matchTemplate benchmark.
#
#   ./run.sh              # build if needed, then run
#   ./run.sh --fresh      # wipe and rebuild both opencv trees first
#   ./run.sh --clean-new  # wipe and rebuild only the new tree first

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PYTHON="${PYTHON:-$(command -v python3)}"

FRESH=0
CLEAN_NEW=0
case "${1:-}" in
  --fresh)     FRESH=1 ;;
  --clean-new) CLEAN_NEW=1 ;;
  "")          ;;
  *)           echo "unknown arg: $1" >&2; exit 2 ;;
esac

find_cv2_path() {
  local build="$1"
  if [[ -d "$build/python_loader" ]] && \
     PYTHONPATH="$build/python_loader" "$PYTHON" -c 'import cv2' >/dev/null 2>&1; then
    echo "$build/python_loader"; return 0
  fi
  local so
  so="$(find "$build" -maxdepth 6 -type f \
        \( -name 'cv2*.so' -o -name 'cv2*.dylib' -o -name 'cv2*.pyd' \) \
        2>/dev/null | head -n1)"
  if [[ -n "$so" ]] && \
     PYTHONPATH="$(dirname "$so")" "$PYTHON" -c 'import cv2' >/dev/null 2>&1; then
    dirname "$so"; return 0
  fi
  return 1
}

ORIG_BUILD="$ROOT/original_opencv/opencv/build"
NEW_BUILD="$ROOT/new_opencv/opencv/build"

if [[ "$FRESH" -eq 1 ]]; then
  ORIG_PATH=""; NEW_PATH=""
else
  ORIG_PATH="$(find_cv2_path "$ORIG_BUILD" || true)"
  NEW_PATH="$(find_cv2_path "$NEW_BUILD"  || true)"
fi
[[ "$CLEAN_NEW" -eq 1 ]] && NEW_PATH=""

build_targets=()
[[ -z "$ORIG_PATH" ]] && build_targets+=("original")
[[ -z "$NEW_PATH"  ]] && build_targets+=("new")

if [[ ${#build_targets[@]} -gt 0 ]]; then
  echo "==> need to build: ${build_targets[*]}"
  build_args=("${build_targets[@]}")
  [[ "$FRESH" -eq 1 ]] && build_args+=("--fresh")
  [[ "$CLEAN_NEW" -eq 1 ]] && build_args+=("--clean-new")
  PYTHON="$PYTHON" "$HERE/build_opencvs.sh" "${build_args[@]}"
  ORIG_PATH="$(find_cv2_path "$ORIG_BUILD" || true)"
  NEW_PATH="$(find_cv2_path "$NEW_BUILD"  || true)"
fi

if [[ -z "$ORIG_PATH" || -z "$NEW_PATH" ]]; then
  echo "!! still missing a cv2 build (orig='$ORIG_PATH' new='$NEW_PATH')" >&2
  exit 1
fi

echo "==> original cv2: $ORIG_PATH"
echo "==> new      cv2: $NEW_PATH"
echo

exec "$PYTHON" "$HERE/masked_template_match.py" \
  --original-cv2 "$ORIG_PATH" \
  --new-cv2      "$NEW_PATH"
