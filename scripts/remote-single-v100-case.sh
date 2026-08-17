#!/usr/bin/env bash
set -euo pipefail

case_name=${1:?case name required}
bin_variant=${2:?binary variant required: base|safe|optimized}
shift 2

root=${SINGLE_V100_RESULT_ROOT:-/home/wzu/models/Qwen3.8-27B-GGUF/single-v100-ab-20260817}
model_root=${MODEL_ROOT:-/home/wzu/models/Qwen3.8-27B-GGUF}
repo_root=${REPO_ROOT:-$HOME/codex-build/dual-v100-llama.cpp/repo}
bin_root=${BIN_ROOT:-$model_root/operator-ab-20260816/$bin_variant/bin}
container=${CONTAINER_NAME:-qwen38-q4-single-ab}
n_predict=${N_PREDICT:-384}
runs=${RUNS:-2}
warmup=${WARMUP:-1}
batch=${BATCH:-2048}
ubatch=${UBATCH:-512}
kv_type=${KV_TYPE:-q8_0}
context=${CONTEXT:-262144}
mmproj=${MMPROJ:-0}
prompt_file=${PROMPT_FILE:-$repo_root/benchmarks/prompts/short.txt}
result_label=${RESULT_LABEL:-short}
docker_gpus=${DOCKER_GPUS:-device=0}
split_mode=${SPLIT_MODE:-none}
main_gpu=${MAIN_GPU:-0}
tensor_split=${TENSOR_SPLIT:-1,1}

mmproj_args=()
if [[ "$mmproj" == 1 ]]; then
    mmproj_args=(--mmproj /models/mmproj-F16.gguf)
fi

# Forward opt-in llama.cpp experiments without baking them into the image.
# Empty values intentionally mean "disabled" so control and candidate cases can
# use the same script and differ only in their exported environment.
docker_env_args=()
for env_name in LLAMA_SPEC_CHAIN LLAMA_SPEC_CHAIN_SUB LLAMA_SCHED_POOL LLAMA_MTP_SUBVOCAB; do
    if [[ -n "${!env_name:-}" ]]; then
        docker_env_args+=(-e "$env_name=${!env_name}")
    fi
done

split_args=(--split-mode "$split_mode" --main-gpu "$main_gpu")
if [[ "$split_mode" == tensor || "$split_mode" == layer || "$split_mode" == row ]]; then
    split_args+=(--tensor-split "$tensor_split")
fi

mkdir -p "$root"
docker rm -f "$container" >/dev/null 2>&1 || true

cleanup() {
    if [[ -n "${monitor_pid:-}" ]]; then
        kill "$monitor_pid" >/dev/null 2>&1 || true
        wait "$monitor_pid" >/dev/null 2>&1 || true
    fi
    docker rm -f "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -d --name "$container" \
    --gpus "$docker_gpus" \
    -p 127.0.0.1:8000:8000 \
    --shm-size 4g \
    -e LD_LIBRARY_PATH=/opt/llama/bin:/usr/local/cuda/lib64 \
    "${docker_env_args[@]}" \
    -v "$bin_root:/opt/llama/bin:ro" \
    -v "$model_root:/models:ro" \
    --entrypoint /opt/llama/bin/llama-server \
    nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 \
    -m /models/Qwen3.8-27B-Q4_K_M.gguf \
    --alias qwen3.8-27b-q4-single \
    --host 0.0.0.0 --port 8000 \
    -ngl all "${split_args[@]}" --fit off \
    -c "$context" --parallel 1 -b "$batch" -ub "$ubatch" \
    -fa on -ctk "$kv_type" -ctv "$kv_type" \
    --jinja --reasoning-format deepseek \
    "${mmproj_args[@]}" \
    "$@" > "$root/$case_name.container-id"

for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:8000/health > "$root/$case_name.health.json"; then
        printf 'case=%s ready_after=%ss\n' "$case_name" "$((i * 2))"
        break
    fi
    if ! docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null | grep -q true; then
        echo "case=$case_name failed_to_load" >&2
        docker logs "$container" 2>&1 | tee "$root/$case_name.log" >&2
        exit 1
    fi
    if [[ $i -eq 120 ]]; then
        echo "case=$case_name health_timeout" >&2
        docker logs "$container" 2>&1 | tee "$root/$case_name.log" >&2
        exit 1
    fi
    sleep 2
done

(
    while docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null | grep -q true; do
        printf '%s,' "$(date +%s)"
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem \
            --format=csv,noheader,nounits | sed -n '1p'
        sleep 1
    done
) > "$root/$case_name.gpu.csv" &
monitor_pid=$!

python3 "$repo_root/benchmarks/llama_native_bench.py" \
    --url http://127.0.0.1:8000 \
    --prompt-file "$prompt_file" \
    --n-predict "$n_predict" --warmup "$warmup" --runs "$runs" \
    --temperature 0 --seed 1 \
    --output "$root/$case_name.$result_label.jsonl" \
    | tee "$root/$case_name.$result_label.out"

kill "$monitor_pid" >/dev/null 2>&1 || true
wait "$monitor_pid" >/dev/null 2>&1 || true
unset monitor_pid
docker logs "$container" > "$root/$case_name.log" 2>&1

awk -F, '$2 ~ /^0$/ {
    gsub(/ /, "", $3); if ($3 + 0 > mem) mem = $3 + 0
    gsub(/ /, "", $4); if ($4 + 0 > util) util = $4 + 0
    gsub(/ /, "", $6); if ($6 + 0 > power) power = $6 + 0
} END {
    printf "case=%s gpu_peak_mem_mib=%d gpu_peak_util_pct=%d gpu_peak_power_w=%.2f\n", name, mem, util, power
}' name="$case_name" "$root/$case_name.gpu.csv"

grep 'draft acceptance' "$root/$case_name.log" | tail -1 || true
