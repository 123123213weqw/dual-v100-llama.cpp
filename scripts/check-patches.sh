#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$ROOT/upstream.lock"

TMP=$(mktemp -d "${TMPDIR:-/tmp}/dual-v100-patch-check.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

if [[ -n "${LLAMA_CPP_MIRROR:-}" ]]; then
    if ! git -C "$LLAMA_CPP_MIRROR" rev-parse --git-dir >/dev/null 2>&1; then
        echo "LLAMA_CPP_MIRROR is not a git repository: $LLAMA_CPP_MIRROR" >&2
        exit 1
    fi
    git clone -q --no-hardlinks --no-checkout "$LLAMA_CPP_MIRROR" "$TMP/llama.cpp"
else
    git init -q "$TMP/llama.cpp"
    git -C "$TMP/llama.cpp" remote add origin "$LLAMA_CPP_REPO"
    git -C "$TMP/llama.cpp" fetch -q --depth=1 origin "$LLAMA_CPP_COMMIT"
fi
git -C "$TMP/llama.cpp" checkout -q --detach "$LLAMA_CPP_COMMIT"

for patch in "$ROOT/patches/safe.patch" "$ROOT/patches/operator.patch"; do
    git -C "$TMP/llama.cpp" reset -q --hard "$LLAMA_CPP_COMMIT"
    git -C "$TMP/llama.cpp" apply --check "$patch"
    echo "OK: $(basename "$patch")"
done

# The MTP sub-vocabulary patch is a composition layer on top of operator.
git -C "$TMP/llama.cpp" reset -q --hard "$LLAMA_CPP_COMMIT"
git -C "$TMP/llama.cpp" apply "$ROOT/patches/operator.patch"
git -C "$TMP/llama.cpp" apply --check "$ROOT/patches/mtp-subvocab.patch"
echo "OK: operator.patch + mtp-subvocab.patch"

git -C "$TMP/llama.cpp" apply "$ROOT/patches/mtp-subvocab.patch"
git -C "$TMP/llama.cpp" apply --check "$ROOT/patches/sm70-tuning.patch"
echo "OK: operator.patch + mtp-subvocab.patch + sm70-tuning.patch"
