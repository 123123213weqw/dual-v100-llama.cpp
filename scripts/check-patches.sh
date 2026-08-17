#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$ROOT/upstream.lock"

TMP=$(mktemp -d "${TMPDIR:-/tmp}/dual-v100-patch-check.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

git init -q "$TMP/llama.cpp"
git -C "$TMP/llama.cpp" remote add origin "$LLAMA_CPP_REPO"
git -C "$TMP/llama.cpp" fetch -q --depth=1 origin "$LLAMA_CPP_COMMIT"
git -C "$TMP/llama.cpp" checkout -q --detach FETCH_HEAD

for patch in "$ROOT/patches/safe.patch" "$ROOT/patches/operator.patch"; do
    git -C "$TMP/llama.cpp" reset -q --hard FETCH_HEAD
    git -C "$TMP/llama.cpp" apply --check "$patch"
    echo "OK: $(basename "$patch")"
done
