#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$ROOT/upstream.lock"

VARIANT=${1:-operator}
DEST=${2:-"$ROOT/.work/llama.cpp-$VARIANT"}

PATCHES=()
case "$VARIANT" in
    baseline) ;;
    safe) PATCHES+=("$ROOT/patches/safe.patch") ;;
    operator) PATCHES+=("$ROOT/patches/operator.patch") ;;
    operator-subvocab)
        PATCHES+=("$ROOT/patches/operator.patch" "$ROOT/patches/mtp-subvocab.patch")
        ;;
    operator-subvocab-tuning)
        PATCHES+=(
            "$ROOT/patches/operator.patch"
            "$ROOT/patches/mtp-subvocab.patch"
            "$ROOT/patches/sm70-tuning.patch"
        )
        ;;
    *)
        echo "usage: $0 {baseline|safe|operator|operator-subvocab|operator-subvocab-tuning} [destination]" >&2
        exit 2
        ;;
esac

if [[ -e "$DEST" ]]; then
    echo "destination already exists: $DEST" >&2
    echo "use a new destination to preserve reproducibility" >&2
    exit 1
fi

mkdir -p "$(dirname "$DEST")"
if [[ -n "${LLAMA_CPP_MIRROR:-}" ]]; then
    if ! git -C "$LLAMA_CPP_MIRROR" rev-parse --git-dir >/dev/null 2>&1; then
        echo "LLAMA_CPP_MIRROR is not a git repository: $LLAMA_CPP_MIRROR" >&2
        exit 1
    fi
    git clone --no-hardlinks --no-checkout "$LLAMA_CPP_MIRROR" "$DEST"
else
    git clone --filter=blob:none --no-checkout "$LLAMA_CPP_REPO" "$DEST"
fi
if ! git -C "$DEST" cat-file -e "$LLAMA_CPP_COMMIT^{commit}" 2>/dev/null; then
    git -C "$DEST" fetch --depth=1 origin "$LLAMA_CPP_COMMIT"
fi
git -C "$DEST" checkout --detach "$LLAMA_CPP_COMMIT"

for patch in "${PATCHES[@]}"; do
    git -C "$DEST" apply --check "$patch"
    git -C "$DEST" apply "$patch"
done

patch_sha256() {
    if [[ ${#PATCHES[@]} -eq 0 ]]; then
        printf 'none'
    else
        local patch
        for patch in "${PATCHES[@]}"; do
            if command -v sha256sum >/dev/null 2>&1; then
                sha256sum "$patch" | awk '{print $1}'
            else
                shasum -a 256 "$patch" | awk '{print $1}'
            fi
        done | paste -sd, -
    fi
}

cat > "$DEST/.dual-v100-build" <<EOF
variant=$VARIANT
upstream=$LLAMA_CPP_COMMIT
patch_sha256=$(patch_sha256)
EOF

echo "prepared $VARIANT at $DEST"
git -C "$DEST" status --short
