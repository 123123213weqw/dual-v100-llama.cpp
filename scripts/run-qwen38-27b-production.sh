#!/usr/bin/env bash
set -euo pipefail
NAME=qwen38-27b
STAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_NAME="${NAME}-pre-kv-${STAMP}"
CACHE=/data/wzu/llama-kv
BACKUPS="$CACHE/deploy-backups"
mkdir -p "$CACHE" "$BACKUPS"
chmod 700 "$CACHE" "$BACKUPS"

# Never interrupt an in-flight inference during a controlled deployment.
for _ in $(seq 1 3600); do
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then break; fi
  BUSY=$(python3 - <<'PY' || echo 1
import json, urllib.request
slot=json.load(urllib.request.urlopen('http://127.0.0.1:8000/slots',timeout=3))[0]
print(1 if slot.get('is_processing') else 0)
PY
)
  [ "$BUSY" = 0 ] && break
  sleep 1
done
[ "${BUSY:-0}" = 0 ] || { echo "slot remained busy; deployment aborted" >&2; exit 75; }

sudo systemctl stop qwen38-kv-manager.service 2>/dev/null || true
OLD_ID=$(docker ps -aqf "name=^${NAME}$")
if [ -n "$OLD_ID" ]; then
  docker inspect "$OLD_ID" > "$BACKUPS/${NAME}-${STAMP}.inspect.json"
  docker rename "$NAME" "$BACKUP_NAME"
  docker update --restart=no "$BACKUP_NAME" >/dev/null
  docker stop -t 30 "$BACKUP_NAME" >/dev/null
fi

rollback() {
  code=$?
  echo "new container failed; rolling back" >&2
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  if [ -n "${OLD_ID:-}" ]; then
    docker rename "$BACKUP_NAME" "$NAME" >/dev/null 2>&1 || true
    docker update --restart=unless-stopped "$NAME" >/dev/null 2>&1 || true
    docker start "$NAME" >/dev/null 2>&1 || true
  fi
  exit "$code"
}
trap rollback ERR

docker run -d \
  --name "$NAME" \
  --restart unless-stopped \
  --gpus all \
  --network host \
  --ipc host \
  --security-opt label=disable \
  --health-cmd='timeout 5 bash -c "</dev/tcp/127.0.0.1/8000"' \
  --health-interval=30s \
  --health-timeout=10s \
  --health-start-period=180s \
  --health-retries=3 \
  --label com.wzu.qwen.engine=v100-x1-meta-tp-fixes-e5b1b578-kv-persist \
  -e GGML_META_ASYNC_HOST_COPY=1 \
  -e LLAMA_EXPERIMENTAL_TP_BACKEND_SAMPLING=1 \
  -e LLAMA_EXPERIMENTAL_TP_LOCAL_TOPK=1 \
  -e LLAMA_EXPERIMENTAL_TP_LOCAL_TOPK_K=10 \
  -e LLAMA_EXPERIMENTAL_MTP_MIRROR_LAYER=0 \
  -e LLAMA_MTP_DEFER_VERIFY=1 \
  -e LLAMA_MTP_DEFER_VERIFY_MIN_POS=32768 \
  -e LLAMA_MTP_CANDIDATE_K=10 \
  -e LLAMA_MTP_ALIGN_TARGET_SAMPLER=1 \
  -e GGML_META_PARALLEL_LAUNCH=1 \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e OMP_NUM_THREADS=16 \
  -e LD_LIBRARY_PATH=/opt/llama/bin:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu \
  -v /home/wzu/models/Qwen3.8-27B-GGUF:/official:ro \
  -v /home/wzu/models/Qwen3.8-27B-Uncensored-GGUF:/uncensored:ro \
  -v /home/wzu/v100-x1/releases/20260829-meta-tp-fixes-e5b1b578/bin:/opt/llama/bin:ro \
  -v "$CACHE":/kv-cache \
  nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 \
  /opt/llama/bin/llama-server \
  -m /uncensored/Qwen3.8-27B-Uncensored-Q8_0.gguf \
  --alias qwen3.8-27b-uncensored \
  --host 0.0.0.0 --port 8000 --metrics \
  -ngl all --split-mode tensor --tensor-split 1,1 --fit off \
  -c 262144 --parallel 1 -b 4096 -ub 2048 \
  -fa on -ctk f16 -ctv f16 \
  --jinja --reasoning-format deepseek \
  --mmproj /uncensored/Qwen3.8-27B-Uncensored-vision-f16.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --spec-draft-backend-sampling --backend-sampling \
  --cache-prompt --cache-idle-slots -cram 65536 \
  --slot-save-path /kv-cache >/dev/null

for _ in $(seq 1 360); do
  if curl -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    trap - ERR
    echo "$BACKUP_NAME" > "$BACKUPS/current-rollback-container"
    echo "model ready; rollback container: $BACKUP_NAME"
    exit 0
  fi
  sleep 1
done
false
