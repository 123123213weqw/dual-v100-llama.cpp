#!/usr/bin/env bash
set -euo pipefail

INTERVAL=${INTERVAL:-1}
OUTPUT=${1:-gpu-telemetry.csv}

QUERY=timestamp,index,name,pci.bus_id,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu,utilization.memory

echo "writing GPU telemetry to $OUTPUT" >&2
nvidia-smi \
    --query-gpu="$QUERY" \
    --format=csv \
    --loop="$INTERVAL" > "$OUTPUT"
