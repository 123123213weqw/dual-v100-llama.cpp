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

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_TESTS=ON

cmake --build "$BUILD_DIR" --parallel "$JOBS" \
    --target llama-server test-backend-ops

LD_LIBRARY_PATH="$BUILD_DIR/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$BUILD_DIR/bin/llama-server" --version

echo "build ready: $BUILD_DIR/bin"
