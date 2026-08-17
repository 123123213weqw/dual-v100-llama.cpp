#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$ROOT/upstream.lock"

VARIANT=${1:-operator}
DEST=${2:-"$ROOT/.work/llama.cpp-$VARIANT"}

case "$VARIANT" in
    baseline) PATCH="" ;;
    safe) PATCH="$ROOT/patches/safe.patch" ;;
    operator) PATCH="$ROOT/patches/operator.patch" ;;
    *)
        echo "usage: $0 {baseline|safe|operator} [destination]" >&2
        exit 2
        ;;
esac

if [[ -e "$DEST" ]]; then
    echo "destination already exists: $DEST" >&2
    echo "use a new destination to preserve reproducibility" >&2
    exit 1
fi

mkdir -p "$(dirname "$DEST")"
git clone --filter=blob:none --no-checkout "$LLAMA_CPP_REPO" "$DEST"
git -C "$DEST" fetch --depth=1 origin "$LLAMA_CPP_COMMIT"
git -C "$DEST" checkout --detach "$LLAMA_CPP_COMMIT"

if [[ -n "$PATCH" ]]; then
    git -C "$DEST" apply --check "$PATCH"
    git -C "$DEST" apply "$PATCH"
fi

patch_sha256() {
    if [[ -z "$PATCH" ]]; then
        printf 'none'
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$PATCH" | awk '{print $1}'
    else
        shasum -a 256 "$PATCH" | awk '{print $1}'
    fi
}

cat > "$DEST/.dual-v100-build" <<EOF
variant=$VARIANT
upstream=$LLAMA_CPP_COMMIT
patch_sha256=$(patch_sha256)
EOF

echo "prepared $VARIANT at $DEST"
git -C "$DEST" status --short
