#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-0}
memory_clock=${V100_MEMORY_CLOCK:-877}
graphics_clock=${V100_GRAPHICS_CLOCK:-1380}

sudo -n nvidia-smi -i "$gpu" -pm 1
sudo -n nvidia-smi -i "$gpu" -ac "$memory_clock,$graphics_clock"
nvidia-smi -i "$gpu" \
    --query-gpu=index,name,persistence_mode,clocks.applications.memory,clocks.applications.graphics,power.limit \
    --format=csv,noheader
