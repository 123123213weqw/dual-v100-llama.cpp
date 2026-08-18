#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VARIANT=${1:-operator}
REMOTE=${REMOTE:-WZU_Server}
REMOTE_ROOT=${REMOTE_ROOT:-'~/codex-build/dual-v100-llama.cpp'}
CUDA_IMAGE=${CUDA_IMAGE:-nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04}
REMOTE_CMAKE_ROOT=${REMOTE_CMAKE_ROOT:-/home/wzu/tools/cmake-3.31.6}

case "$VARIANT" in
    baseline) PATCH_SHA=baseline ;;
    safe|operator)
        PATCH_SHA=$(shasum -a 256 "$ROOT/patches/$VARIANT.patch" | awk '{print substr($1,1,12)}')
        ;;
    operator-subvocab|operator-subvocab-tuning)
        PATCH_FILES=("$ROOT/patches/operator.patch" "$ROOT/patches/mtp-subvocab.patch")
        if [[ "$VARIANT" == operator-subvocab-tuning ]]; then
            PATCH_FILES+=("$ROOT/patches/sm70-tuning.patch")
        fi
        PATCH_SHA=$(cat "${PATCH_FILES[@]}" | \
            shasum -a 256 | awk '{print substr($1,1,12)}')
        ;;
    *)
        echo "usage: $0 {baseline|safe|operator|operator-subvocab|operator-subvocab-tuning}" >&2
        exit 2
        ;;
esac

TUNING_SUFFIX=""
if [[ "$VARIANT" == operator-subvocab-tuning ]]; then
    FATTN_THREADS=${GGML_CUDA_FATTN_VEC_NTHREADS:-128}
    FATTN_MIN_BLOCKS=${GGML_CUDA_FATTN_VEC_MIN_BLOCKS:-1}
    FATTN_GQA_HEADS=${GGML_CUDA_FATTN_VEC_GQA_HEADS:-1}
    GDN_WARPS=${GGML_CUDA_GDN_NUM_WARPS:-4}
    GDN_ROWS=${GGML_CUDA_GDN_DV_PER_WARP:-4}
    for pair in \
        "GGML_CUDA_FATTN_VEC_NTHREADS:$FATTN_THREADS" \
        "GGML_CUDA_FATTN_VEC_MIN_BLOCKS:$FATTN_MIN_BLOCKS" \
        "GGML_CUDA_FATTN_VEC_GQA_HEADS:$FATTN_GQA_HEADS" \
        "GGML_CUDA_GDN_NUM_WARPS:$GDN_WARPS" \
        "GGML_CUDA_GDN_DV_PER_WARP:$GDN_ROWS"; do
        name=${pair%%:*}
        value=${pair#*:}
        if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
            echo "$name must be a positive integer, got: $value" >&2
            exit 2
        fi
    done
    if [[ "$FATTN_GQA_HEADS" != 1 && "$FATTN_GQA_HEADS" != 2 ]]; then
        echo "GGML_CUDA_FATTN_VEC_GQA_HEADS must be 1 or 2, got: $FATTN_GQA_HEADS" >&2
        exit 2
    fi
    export GGML_CUDA_FATTN_VEC_NTHREADS=$FATTN_THREADS
    export GGML_CUDA_FATTN_VEC_MIN_BLOCKS=$FATTN_MIN_BLOCKS
    export GGML_CUDA_FATTN_VEC_GQA_HEADS=$FATTN_GQA_HEADS
    export GGML_CUDA_GDN_NUM_WARPS=$GDN_WARPS
    export GGML_CUDA_GDN_DV_PER_WARP=$GDN_ROWS
    TUNING_SUFFIX="-fa${FATTN_THREADS}b${FATTN_MIN_BLOCKS}gqa${FATTN_GQA_HEADS}-gdn${GDN_WARPS}x${GDN_ROWS}"
fi

REMOTE_UPSTREAM_MIRROR=${REMOTE_UPSTREAM_MIRROR:-}
REMOTE_MIRROR_ENV=""
if [[ -n "$REMOTE_UPSTREAM_MIRROR" ]]; then
    REMOTE_MIRROR_ENV="LLAMA_CPP_MIRROR=$REMOTE_UPSTREAM_MIRROR"
fi

REMOTE_ROOT_ABS=$(ssh "$REMOTE" "mkdir -p $REMOTE_ROOT && cd $REMOTE_ROOT && pwd")
REMOTE_REPO="$REMOTE_ROOT_ABS/repo"
REMOTE_SOURCE="$REMOTE_ROOT_ABS/src-$VARIANT-$PATCH_SHA"
REMOTE_BUILD="$REMOTE_ROOT_ABS/build-$VARIANT-$PATCH_SHA$TUNING_SUFFIX"

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
    if [ ! -e $REMOTE_SOURCE ]; then \
        $REMOTE_MIRROR_ENV ./scripts/prepare-source.sh $VARIANT $REMOTE_SOURCE; \
    fi && \
    test -d '$REMOTE_CMAKE_ROOT' && \
    docker run --rm --gpus device=0 \
      -e PATH=/opt/cmake/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
      -e JOBS='${JOBS:-}' \
      -e GGML_CUDA_FATTN_VEC_NTHREADS='${GGML_CUDA_FATTN_VEC_NTHREADS:-}' \
      -e GGML_CUDA_FATTN_VEC_MIN_BLOCKS='${GGML_CUDA_FATTN_VEC_MIN_BLOCKS:-}' \
      -e GGML_CUDA_FATTN_VEC_GQA_HEADS='${GGML_CUDA_FATTN_VEC_GQA_HEADS:-}' \
      -e GGML_CUDA_GDN_NUM_WARPS='${GGML_CUDA_GDN_NUM_WARPS:-}' \
      -e GGML_CUDA_GDN_DV_PER_WARP='${GGML_CUDA_GDN_DV_PER_WARP:-}' \
      -v '$REMOTE_ROOT_ABS:/work' \
      -v '$REMOTE_CMAKE_ROOT:/opt/cmake:ro' \
      -w /work/repo \
      '$CUDA_IMAGE' \
      bash ./scripts/build-sm70.sh \
        '/work/src-$VARIANT-$PATCH_SHA' \
        '/work/build-$VARIANT-$PATCH_SHA$TUNING_SUFFIX'"

echo "remote build: $REMOTE:$REMOTE_BUILD/bin"

if [[ ${FETCH_ARTIFACTS:-0} == 1 ]]; then
    mkdir -p "$ROOT/dist/$VARIANT-$PATCH_SHA"
    rsync -az "$REMOTE:$REMOTE_BUILD/bin/" "$ROOT/dist/$VARIANT-$PATCH_SHA/"
    echo "downloaded: $ROOT/dist/$VARIANT-$PATCH_SHA"
fi
