#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=${CONFIG:-"$ROOT/config/qwen3.8-27b.env.example"}
BUILD_DIR=${BUILD_DIR:-"$ROOT/.work/build-operator-subvocab"}

if [[ ! -f "$CONFIG" ]]; then
    echo "config not found: $CONFIG" >&2
    exit 1
fi

set -a
# shellcheck disable=SC1090
source "$CONFIG"
set +a

: "${MODEL:?MODEL must point to a GGUF file}"
: "${MODEL_ALIAS:=qwen3.8-27b}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${CUDA_VISIBLE_DEVICES:=0,1}"
: "${CONTEXT:=262144}"
: "${PARALLEL:=1}"
: "${BATCH:=2048}"
: "${UBATCH:=512}"
: "${SPLIT_MODE:=tensor}"
: "${MAIN_GPU:=0}"
: "${TENSOR_SPLIT:=1,1}"
: "${KV_TYPE_K:=f16}"
: "${KV_TYPE_V:=f16}"
: "${SPEC_TYPE:=draft-mtp}"
: "${SPEC_DRAFT_MAX:=3}"
: "${SPEC_DRAFT_BACKEND_SAMPLING:=off}"

SERVER="$BUILD_DIR/bin/llama-server"
if [[ ! -x "$SERVER" ]]; then
    echo "llama-server not found: $SERVER" >&2
    exit 1
fi

ARGS=(
    -m "$MODEL"
    --alias "$MODEL_ALIAS"
    --host "$HOST"
    --port "$PORT"
    -ngl all
    --split-mode "$SPLIT_MODE"
    --fit off
    -c "$CONTEXT"
    --parallel "$PARALLEL"
    -b "$BATCH"
    -ub "$UBATCH"
    -fa on
    -ctk "$KV_TYPE_K"
    -ctv "$KV_TYPE_V"
    --jinja
    --reasoning-format deepseek
)

if [[ "$SPLIT_MODE" == tensor ]]; then
    ARGS+=(--tensor-split "$TENSOR_SPLIT")
else
    ARGS+=(--main-gpu "$MAIN_GPU")
fi

if [[ "$SPEC_TYPE" != none ]]; then
    ARGS+=(--spec-type "$SPEC_TYPE" --spec-draft-n-max "$SPEC_DRAFT_MAX")
fi

case "$SPEC_DRAFT_BACKEND_SAMPLING" in
    on)   ARGS+=(--spec-draft-backend-sampling) ;;
    off)  ARGS+=(--no-spec-draft-backend-sampling) ;;
    auto) ;;
    *)
        echo "SPEC_DRAFT_BACKEND_SAMPLING must be on, off, or auto" >&2
        exit 1
        ;;
esac

if [[ -n ${MMPROJ:-} ]]; then
    ARGS+=(--mmproj "$MMPROJ")
fi

export CUDA_VISIBLE_DEVICES
if [[ ${MTP_SUBVOCAB:-0} -gt 0 ]]; then
    export LLAMA_MTP_SUBVOCAB="$MTP_SUBVOCAB"
fi
export LD_LIBRARY_PATH="$BUILD_DIR/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$SERVER" "${ARGS[@]}" "$@"
