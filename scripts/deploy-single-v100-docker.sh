#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
container=${CONTAINER_NAME:-qwen38-27b}
image=${CUDA_IMAGE:-nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04}
model_root=${MODEL_ROOT:-/home/wzu/models/Qwen3.8-27B-GGUF}
bin_root=${BIN_ROOT:-$model_root/operator-subvocab-20260817/bin}
host_port=${HOST_PORT:-8000}
gpu=${GPU:-0}
mtp_subvocab=${MTP_SUBVOCAB:-131072}

docker_env_args=()
if [[ "$mtp_subvocab" -gt 0 ]]; then
    docker_env_args+=(-e "LLAMA_MTP_SUBVOCAB=$mtp_subvocab")
fi

if [[ ${TUNE_CLOCK:-1} == 1 ]]; then
    "$root/scripts/tune-v100-clock.sh" "$gpu"
fi

docker rm -f "$container" >/dev/null 2>&1 || true
docker run -d --name "$container" \
    --restart unless-stopped \
    --gpus "device=$gpu" \
    -p "127.0.0.1:$host_port:8000" \
    --shm-size 4g \
    -e LD_LIBRARY_PATH=/opt/llama/bin:/usr/local/cuda/lib64 \
    "${docker_env_args[@]}" \
    -v "$bin_root:/opt/llama/bin:ro" \
    -v "$model_root:/models:ro" \
    --entrypoint /opt/llama/bin/llama-server \
    "$image" \
    -m /models/Qwen3.8-27B-Q4_K_M.gguf \
    --alias qwen3.8-27b \
    --host 0.0.0.0 --port 8000 \
    -ngl all --split-mode none --main-gpu 0 --fit off \
    -c 262144 --parallel 1 -b 2048 -ub 512 \
    -fa on -ctk q4_0 -ctv q4_0 \
    --jinja --reasoning-format deepseek \
    --mmproj /models/mmproj-F16.gguf \
    --image-min-tokens 1024 \
    --spec-type draft-mtp --spec-draft-n-max 1

for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$host_port/health"; then
        printf '\n%s is ready after %ss\n' "$container" "$((i * 2))"
        exit 0
    fi
    if ! docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null | grep -q true; then
        docker logs "$container" >&2
        exit 1
    fi
    sleep 2
done

docker logs "$container" >&2
exit 1
