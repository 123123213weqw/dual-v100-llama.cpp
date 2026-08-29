#!/usr/bin/env bash
set -Eeuo pipefail

BASE=/home/data/wangyue/repos/codex-build/llama.cpp-dflash2-upstream-standalone-20260829/build-v100-dflash2
TARGET_DIR=/home/wzu/models/Qwen3.8-27B-Uncensored-GGUF
DRAFT_DIR=/home/data/wangyue/models/qwen38-dflash2-q8
KV=/data/wzu/llama-kv
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=/home/wzu/v100-x1/ab-runs/mtp-vs-dflash-$STAMP
TEST_NAME=qwen38-dflash2-test
PROD_STOPPED=0
mkdir -p "$OUT"
echo "$OUT" > /home/wzu/v100-x1/ab-runs/mtp-vs-dflash-current
exec > >(tee -a "$OUT/run.log") 2>&1

cleanup() {
  rc=$?
  set +e
  docker logs "$TEST_NAME" >"$OUT/last-test.log" 2>&1 || true
  docker rm -f "$TEST_NAME" >/dev/null 2>&1 || true
  if [[ "$PROD_STOPPED" == 1 ]]; then
    docker start qwen38-27b >/dev/null 2>&1 || true
    for i in $(seq 1 180); do
      curl -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1 && break
      sleep 1
    done
  fi
  sudo systemctl start qwen38-kv-manager.service >/dev/null 2>&1 || true
  echo "[cleanup] rc=$rc production=$(docker inspect -f '{{.State.Status}} {{.State.Health.Status}}' qwen38-27b 2>/dev/null) kv=$(systemctl is-active qwen38-kv-manager.service 2>/dev/null)"
  exit "$rc"
}
trap cleanup EXIT

echo "[wait] wait for production slot to become idle"
while true; do
  busy=$(curl -fsS --max-time 5 http://127.0.0.1:8000/slots 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(1 if d and d[0].get("is_processing") else 0)' 2>/dev/null || echo 1)
  [[ "$busy" == 0 ]] && break
  sleep 1
done

echo "[checkpoint] save exact pre-benchmark conversation"
sudo systemctl stop qwen38-kv-manager.service
while true; do
  busy=$(curl -fsS --max-time 5 http://127.0.0.1:8000/slots | python3 -c 'import json,sys; d=json.load(sys.stdin); print(1 if d and d[0].get("is_processing") else 0)')
  [[ "$busy" == 0 ]] && break
  sleep 1
done
mkdir -p "$KV/deploy-backups/mtp-vs-dflash-$STAMP"
ln "$KV/latest.bin" "$KV/deploy-backups/mtp-vs-dflash-$STAMP/latest.bin" 2>/dev/null || cp --reflink=auto "$KV/latest.bin" "$KV/deploy-backups/mtp-vs-dflash-$STAMP/latest.bin"
cp -a "$KV/latest.json" "$KV/deploy-backups/mtp-vs-dflash-$STAMP/latest.json"
rm -f "$KV/ab-next.bin"
save_json=$(curl -fsS --max-time 1800 -X POST 'http://127.0.0.1:8000/slots/0?action=save' -H 'Content-Type: application/json' -d '{"filename":"ab-next.bin"}')
printf '%s\n' "$save_json" >"$OUT/checkpoint-save.json"
tokens=$(python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["n_saved"])' <<<"$save_json")
bytes=$(stat -c %s "$KV/ab-next.bin")
mv -f "$KV/latest.bin" "$KV/previous.bin"
mv -f "$KV/ab-next.bin" "$KV/latest.bin"
python3 - "$KV/latest.json" "$tokens" "$bytes" <<'PY'
import datetime,json,sys
p,t,b=sys.argv[1:]
json.dump({"savedAt":datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),"tokens":int(t),"bytes":int(b),"slot":0},open(p,"w"),indent=2)
PY
sync
echo "[checkpoint] tokens=$tokens bytes=$bytes"

run_requests() {
  mode=$1
  mkdir -p "$OUT/$mode"
  python3 - "$mode" "$OUT/$mode" <<'PY'
import json,sys,time,urllib.request
mode,out=sys.argv[1:]
task="审查下面 C 函数的内存安全问题，列出漏洞、触发条件和最小修复代码：\nchar *f(const char *s) { char b[8]; strcpy(b, s); return strdup(b); }"
filler="".join(f"Reference record {i:06d}: memory safety bounds checking integer overflow use-after-free race condition.\n" for i in range(5080))
cases=[
  ("short", [{"role":"user","content":task}]),
  ("long117k", [{"role":"system","content":filler},{"role":"user","content":task}]),
]
for name,messages in cases:
    body={"model":"qwen3.8-27b-uncensored","messages":messages,"temperature":0,"top_p":1,"top_k":1,"seed":12345,"max_tokens":512,"stream":False}
    req=urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",data=json.dumps(body,ensure_ascii=False).encode(),headers={"Content-Type":"application/json"})
    t=time.time()
    with urllib.request.urlopen(req,timeout=3600) as r: data=json.load(r)
    data["_wall_seconds"]=time.time()-t
    json.dump(data,open(f"{out}/{name}.json","w"),ensure_ascii=False,indent=2)
    print(json.dumps({"mode":mode,"case":name,"wall":data["_wall_seconds"],"timings":data.get("timings",{})},ensure_ascii=False),flush=True)
PY
}

echo "[A] current optimized production MTP"
run_requests mtp

echo "[switch] stop production; launch DFlash2 Q8"
docker stop -t 90 qwen38-27b >/dev/null
PROD_STOPPED=1
docker run -d --name "$TEST_NAME" --gpus all --network host \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e LD_LIBRARY_PATH=/opt/build/bin:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu \
  -v "$BASE":/opt/build:ro \
  -v "$TARGET_DIR":/target:ro \
  -v "$DRAFT_DIR":/draft:ro \
  llama-v100-builder:cuda12.8 \
  /opt/build/bin/llama-server \
  -m /target/Qwen3.8-27B-Uncensored-Q8_0.gguf \
  --alias qwen3.8-27b-uncensored --host 127.0.0.1 --port 8000 --metrics \
  -ngl all --split-mode tensor --tensor-split 1,1 --fit off \
  -c 131072 --parallel 1 -b 4096 -ub 2048 -fa on \
  -ctk f16 -ctv f16 --jinja --reasoning-format deepseek --cache-prompt \
  --spec-type draft-dflash \
  --spec-draft-model /draft/Qwen3.8-27B-DFlash2-Q8_0.gguf \
  --spec-draft-ngl all --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \
  --spec-draft-n-max 7 --spec-draft-n-min 1 >/dev/null
for i in $(seq 1 240); do
  curl -fsS --max-time 3 http://127.0.0.1:8000/health >/dev/null 2>&1 && break
  if ! docker inspect -f '{{.State.Running}}' "$TEST_NAME" 2>/dev/null | grep -qx true; then
    docker logs "$TEST_NAME"
    exit 1
  fi
  sleep 1
done
curl -fsS http://127.0.0.1:8000/health >/dev/null
echo "[B] DFlash2 Q8"
run_requests dflash
docker logs "$TEST_NAME" >"$OUT/dflash-server.log" 2>&1 || true
docker rm -f "$TEST_NAME" >/dev/null

python3 - "$OUT" <<'PY'
import json,statistics,sys
out=sys.argv[1]
rows=[]
for name in ["short","long117k"]:
    a=json.load(open(f"{out}/mtp/{name}.json")); b=json.load(open(f"{out}/dflash/{name}.json"))
    ta=a.get("timings",{}); tb=b.get("timings",{})
    def text(x):
        m=x["choices"][0]["message"]
        return (m.get("reasoning_content") or "")+"\n<FINAL>\n"+(m.get("content") or "")
    row={
      "case":name,
      "mtp_prompt_n":ta.get("prompt_n"),"dflash_prompt_n":tb.get("prompt_n"),
      "mtp_prompt_tps":ta.get("prompt_per_second"),"dflash_prompt_tps":tb.get("prompt_per_second"),
      "mtp_decode_tps":ta.get("predicted_per_second"),"dflash_decode_tps":tb.get("predicted_per_second"),
      "mtp_draft_n":ta.get("draft_n"),"mtp_accepted":ta.get("draft_n_accepted"),
      "dflash_draft_n":tb.get("draft_n"),"dflash_accepted":tb.get("draft_n_accepted"),
      "mtp_wall":a["_wall_seconds"],"dflash_wall":b["_wall_seconds"],
      "exact_match":text(a)==text(b),
    }
    row["decode_ratio_dflash_over_mtp"]=row["dflash_decode_tps"]/row["mtp_decode_tps"]
    rows.append(row)
summary={"rows":rows}
json.dump(summary,open(f"{out}/summary.json","w"),ensure_ascii=False,indent=2)
print(json.dumps(summary,ensure_ascii=False,indent=2))
PY

echo "[done] fair A/B complete"
