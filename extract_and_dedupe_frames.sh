#!/usr/bin/env bash
# Extract MP4 frames, then deduplicate nearly identical consecutive frames.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv_extract/bin/python" ]]; then
    PYTHON="$ROOT/.venv_extract/bin/python"
  else
    PYTHON="$(command -v python3)"
  fi
fi

default_frame_out="$HERE/data/images"
frame_out="${EXTRACT_FRAME_OUT:-}"

if [[ -z "$frame_out" ]]; then
  i=1
  while [[ $i -le $# ]]; do
    a="${!i}"
    if [[ "$a" == "--out" ]]; then
      n=$((i + 1))
      if [[ $n -le $# ]]; then
        frame_out="${!n}"
      fi
      break
    fi
    if [[ "$a" == --out=* ]]; then
      frame_out="${a#--out=}"
      break
    fi
    i=$((i + 1))
  done
fi

[[ -z "$frame_out" ]] && frame_out="$default_frame_out"
DEDUP_OUT="${DEDUP_OUT:-$HERE/data/images_deduped}"

echo "==> extract -> $frame_out"
"$PYTHON" "$HERE/extract_mp4_frames.py" "$@"

echo
echo "==> dedupe  (from $frame_out) -> $DEDUP_OUT"
"$PYTHON" "$HERE/dedupe_similar_frames.py" --in "$frame_out" --out "$DEDUP_OUT"
