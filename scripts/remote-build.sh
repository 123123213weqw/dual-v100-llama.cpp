#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VARIANT=${1:-operator}
REMOTE=${REMOTE:-WZU_Server}
REMOTE_ROOT=${REMOTE_ROOT:-'~/codex-build/dual-v100-llama.cpp'}

case "$VARIANT" in
    baseline) PATCH_SHA=baseline ;;
    safe|operator)
        PATCH_SHA=$(shasum -a 256 "$ROOT/patches/$VARIANT.patch" | awk '{print substr($1,1,12)}')
        ;;
    operator-subvocab)
        PATCH_SHA=$(cat "$ROOT/patches/operator.patch" "$ROOT/patches/mtp-subvocab.patch" | \
            shasum -a 256 | awk '{print substr($1,1,12)}')
        ;;
    *)
        echo "usage: $0 {baseline|safe|operator|operator-subvocab}" >&2
        exit 2
        ;;
esac

REMOTE_REPO="$REMOTE_ROOT/repo"
REMOTE_SOURCE="$REMOTE_ROOT/src-$VARIANT-$PATCH_SHA"
REMOTE_BUILD="$REMOTE_ROOT/build-$VARIANT-$PATCH_SHA"

ssh "$REMOTE" "mkdir -p $REMOTE_REPO"
rsync -az --delete \
    --exclude='.git/' \
    --exclude='.work/' \
    --exclude='build/' \
    --exclude='.env' \
    --exclude='.env.*' \
    --exclude='results/raw/' \
    "$ROOT/" "$REMOTE:$REMOTE_REPO/"

ssh "$REMOTE" "cd $REMOTE_REPO && \
    if [ ! -e $REMOTE_SOURCE ]; then ./scripts/prepare-source.sh $VARIANT $REMOTE_SOURCE; fi && \
    ./scripts/build-sm70.sh $REMOTE_SOURCE $REMOTE_BUILD"

echo "remote build: $REMOTE:$REMOTE_BUILD/bin"

if [[ ${FETCH_ARTIFACTS:-0} == 1 ]]; then
    mkdir -p "$ROOT/dist/$VARIANT-$PATCH_SHA"
    rsync -az "$REMOTE:$REMOTE_BUILD/bin/" "$ROOT/dist/$VARIANT-$PATCH_SHA/"
    echo "downloaded: $ROOT/dist/$VARIANT-$PATCH_SHA"
fi
