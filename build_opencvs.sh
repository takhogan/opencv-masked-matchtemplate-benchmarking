#!/usr/bin/env bash
# Build cv2 from original_opencv and/or new_opencv and print the resulting
# cv2 package paths. Does not run any benchmarks — use ./run.sh for that.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$(command -v python3)}"
if [[ -z "${JOBS:-}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
  elif command -v sysctl >/dev/null 2>&1 && sysctl -n hw.ncpu >/dev/null 2>&1; then
    JOBS="$(sysctl -n hw.ncpu)"
  else
    JOBS=4
  fi
fi

if [[ -z "${WITH_CUDA:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then WITH_CUDA=ON; else WITH_CUDA=OFF; fi
fi

# Locate the CUDA driver stub (libcuda.so) when building with CUDA. On systems
# without a real driver (e.g. Colab build hosts) the stub lives under
# $CUDA_HOME/lib64/stubs and isn't on the default search path, so cudacodec
# fails to configure with CUDA_CUDA_LIBRARY-NOTFOUND. Pass it explicitly when
# we find it; otherwise leave CMake to its own defaults.
CUDA_STUB_FLAG=()
if [[ "$WITH_CUDA" == "ON" ]]; then
  # First check a real driver lib (present on hosts with an Nvidia driver),
  # then fall back to the stub shipped with the CUDA toolkit.
  cuda_lib=""
  for cand in \
      "${CUDA_CUDA_LIBRARY:-}" \
      /usr/lib/x86_64-linux-gnu/libcuda.so \
      /usr/lib64/libcuda.so \
      /usr/lib/libcuda.so \
      "${CUDA_HOME:-}/lib64/stubs/libcuda.so" \
      "${CUDA_PATH:-}/lib64/stubs/libcuda.so" \
      /usr/local/cuda/lib64/stubs/libcuda.so \
      /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so \
      /usr/lib/x86_64-linux-gnu/stubs/libcuda.so; do
    if [[ -n "$cand" && -e "$cand" ]]; then cuda_lib="$cand"; break; fi
  done
  # Last resort: scan common CUDA install roots.
  if [[ -z "$cuda_lib" ]]; then
    cuda_lib="$(find /usr/local/cuda* /opt/cuda* /usr/lib /usr/lib64 \
                  -maxdepth 6 -name 'libcuda.so' 2>/dev/null | head -n1)"
  fi
  if [[ -n "$cuda_lib" ]]; then
    CUDA_STUB_FLAG=(-DCUDA_CUDA_LIBRARY="$cuda_lib")
    echo "==> CUDA driver lib: $cuda_lib"
  else
    echo "!! WITH_CUDA=ON but no libcuda.so found; disabling cudacodec" >&2
    CUDA_STUB_FLAG=(
      -DBUILD_opencv_cudacodec=OFF
      -DWITH_NVCUVID=OFF
      -DWITH_NVCUVENC=OFF
    )
  fi
fi

# What to build.
TARGETS=()
FRESH=0
CLEAN_NEW=0
TARGETS_EXPLICIT=0
for arg in "$@"; do
  case "$arg" in
    original|new) TARGETS+=("$arg"); TARGETS_EXPLICIT=1 ;;
    --fresh) FRESH=1 ;;
    --clean-new) CLEAN_NEW=1 ;;
    -h|--help)
      sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# Only build the OpenCV modules the benchmark actually uses. The benchmark
# exercises CPU matchTemplate (imgproc), OpenCL matchTemplate (UMat path,
# also in imgproc/core), and CUDA matchTemplate (cudaimgproc). imgcodecs is
# needed for imread/imwrite, python3 for the bindings. BUILD_LIST resolves
# transitive deps automatically.
BUILD_LIST="core,imgproc,imgcodecs,python3"
if [[ "$WITH_CUDA" == "ON" ]]; then
  # cudev is required by core whenever WITH_CUDA=ON (it lives in opencv_contrib).
  BUILD_LIST="$BUILD_LIST,cudev,cudaimgproc"
fi
# --clean-new implies "build new" unless the user explicitly listed targets.
if [[ "$CLEAN_NEW" -eq 1 && "$TARGETS_EXPLICIT" -eq 0 ]]; then
  TARGETS=(new)
fi
if [[ ${#TARGETS[@]} -eq 0 ]]; then TARGETS=(original new); fi

echo "==> python  : $PYTHON"
echo "==> jobs    : $JOBS"
echo "==> CUDA    : $WITH_CUDA"
echo "==> targets : ${TARGETS[*]}"
echo "==> modules : $BUILD_LIST"
echo

PY_INC="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
PY_LIB="$("$PYTHON" -c 'import sysconfig,glob,os; \
libdir=sysconfig.get_config_var("LIBDIR"); \
ldlib=sysconfig.get_config_var("LDLIBRARY"); \
print(os.path.join(libdir, ldlib))')"
NUMPY_INC="$("$PYTHON" -c 'import numpy; print(numpy.get_include())')"

build_one() {
  local tag="$1"   # "original" or "new"
  local tree
  case "$tag" in
    original) tree="$ROOT/original_opencv" ;;
    new)      tree="$ROOT/new_opencv" ;;
    *) echo "bad tag $tag" >&2; return 1 ;;
  esac

  local src="$tree/opencv"
  local contrib="$tree/opencv_contrib/modules"
  local build="$src/build"

  if [[ ! -d "$src" ]]; then
    echo "!! missing $src — skipping $tag" >&2; return 0
  fi
  if [[ ! -d "$contrib" ]]; then
    echo "!! missing $contrib — skipping $tag" >&2; return 0
  fi

  local wipe=0
  [[ "$FRESH" -eq 1 ]] && wipe=1
  [[ "$CLEAN_NEW" -eq 1 && "$tag" == "new" ]] && wipe=1
  if [[ "$wipe" -eq 1 && -d "$build" ]]; then
    echo "==> [$tag] clean: removing $build"
    rm -rf "$build"
  fi

  echo "==> [$tag] configuring in $build"
  mkdir -p "$build"
  (
    cd "$build"
    cmake "$src" \
      -DCMAKE_BUILD_TYPE=Release \
      -DOPENCV_EXTRA_MODULES_PATH="$contrib" \
      -DWITH_OPENCL=ON \
      -DWITH_CUDA="$WITH_CUDA" \
      -DBUILD_opencv_python3=ON \
      -DBUILD_opencv_python_bindings_generator=ON \
      -DPYTHON3_EXECUTABLE="$PYTHON" \
      -DPYTHON3_INCLUDE_DIR="$PY_INC" \
      -DPYTHON3_LIBRARY="$PY_LIB" \
      -DPYTHON3_NUMPY_INCLUDE_DIRS="$NUMPY_INC" \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_DOCS=OFF \
      -DBUILD_JAVA=OFF \
      -DBUILD_opencv_apps=OFF \
      -DCMAKE_C_FLAGS="-w" \
      -DCMAKE_CXX_FLAGS="-w" \
      ${CUDA_STUB_FLAG[@]+"${CUDA_STUB_FLAG[@]}"} \
      -DBUILD_LIST="$BUILD_LIST" \
      ${CMAKE_EXTRA:-}

    echo "==> [$tag] building (-j$JOBS)"
    cmake --build . -j "$JOBS" --target opencv_python3 2>&1 \
      | awk '
          /: note: candidate (template|function|constructor) ignored/ { skip=3 }
          skip>0 { skip--; next }
          { print }
        '
  )
}

find_cv2_path() {
  local build="$1"
  if [[ -d "$build/python_loader" ]]; then
    echo "$build/python_loader"; return 0
  fi
  local so
  so="$(find "$build" -maxdepth 6 -type f \
        \( -name 'cv2*.so' -o -name 'cv2*.dylib' -o -name 'cv2*.pyd' \) \
        2>/dev/null | head -n1)"
  if [[ -n "$so" ]]; then dirname "$so"; return 0; fi
  return 1
}

for tag in "${TARGETS[@]}"; do
  build_one "$tag"
done

echo
echo "==> locating cv2 packages"
for tag in "${TARGETS[@]}"; do
  local_build=""
  case "$tag" in
    original) local_build="$ROOT/original_opencv/opencv/build" ;;
    new)      local_build="$ROOT/new_opencv/opencv/build" ;;
  esac
  if path="$(find_cv2_path "$local_build")"; then
    echo "    $tag -> $path"
    PYTHONPATH="$path" "$PYTHON" -c \
      "import cv2; print('       ok: cv2', cv2.__version__, 'from', cv2.__file__)" \
      || echo "       !! import check failed"
  else
    echo "    $tag -> (not found under $local_build)"
  fi
done

echo
echo "==> done. Run ./run.sh to execute the benchmark."
