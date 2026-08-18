#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_DIR=${1:-"$ROOT/.work/llama.cpp-operator"}
BUILD_DIR=${2:-"$ROOT/.work/build-operator"}

if [[ ! -f "$SOURCE_DIR/CMakeLists.txt" ]]; then
    echo "llama.cpp source not found: $SOURCE_DIR" >&2
    exit 1
fi

if command -v nproc >/dev/null 2>&1; then
    DEFAULT_JOBS=$(nproc)
else
    DEFAULT_JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)
fi
JOBS=${JOBS:-$DEFAULT_JOBS}

CUDA_FLAGS=${CMAKE_CUDA_FLAGS:-}
append_cuda_define() {
    local name=$1
    local value=${!name:-}
    if [[ -z "$value" ]]; then
        return
    fi
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer, got: $value" >&2
        exit 2
    fi
    CUDA_FLAGS+=" -D${name}=${value}"
}

append_cuda_define GGML_CUDA_FATTN_VEC_NTHREADS
append_cuda_define GGML_CUDA_FATTN_VEC_MIN_BLOCKS
append_cuda_define GGML_CUDA_FATTN_VEC_GQA_HEADS
append_cuda_define GGML_CUDA_GDN_NUM_WARPS
append_cuda_define GGML_CUDA_GDN_DV_PER_WARP

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CUDA_ARCHITECTURES=70
    -DGGML_CUDA=ON
    -DGGML_NATIVE=OFF
    -DBUILD_SHARED_LIBS=ON
    -DLLAMA_BUILD_TESTS=ON
)
if [[ -n "${CUDA_FLAGS// }" ]]; then
    CMAKE_ARGS+=("-DCMAKE_CUDA_FLAGS=$CUDA_FLAGS")
fi

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" \
    "${CMAKE_ARGS[@]}"

cmake --build "$BUILD_DIR" --parallel "$JOBS" \
    --target llama-server test-backend-ops

LD_LIBRARY_PATH="$BUILD_DIR/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$BUILD_DIR/bin/llama-server" --version

echo "build ready: $BUILD_DIR/bin"
