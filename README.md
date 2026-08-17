# dual-v100-llama.cpp

Reproducible `llama.cpp` kernel and runtime optimization lab for **two NVIDIA
Tesla V100 GPUs (SM70)**. The primary workload is single-user, long-context
decode with tensor parallelism; Qwen3.8-27B is the first validated model.

This repository is intentionally a thin patch and benchmark layer rather than
a permanent fork. It pins an upstream `llama.cpp` revision, carries small
reviewable patch variants, and makes every performance claim reproducible.

[中文实验报告](docs/benchmark-2026-08-16.zh-CN.md)

## Current result

Validated on 2x V100 with tensor split `1,1`, context `262144`, parallel `1`,
batch `2048`, ubatch `512`, Flash Attention enabled, F16 KV cache, and MTP draft
length 3.

| Variant | Short decode | 71K decode | 71K prefill | Memory / GPU |
|---|---:|---:|---:|---:|
| upstream baseline | 57.400 tok/s | 61.747 tok/s | 655.332 tok/s | 23,735 MiB |
| safe | 57.303 tok/s | - | - | 23,665 MiB |
| operator | **57.677 tok/s** | **62.421 tok/s** | 652.111 tok/s | 23,735 MiB |

The operator variant improves decode by `0.48%` on the short case and `1.09%`
at 71K context. Q8_0 KV saves about `2.8 GiB` per GPU but is slower at long
context, so F16 KV remains the production default. Raw summary data lives in
[`results/summary-2026-08-16.tsv`](results/summary-2026-08-16.tsv).

## Patch variants

- [`patches/safe.patch`](patches/safe.patch)
  - multi-block CUDA ARGMAX for very wide rows;
  - serialization of multi-ubatch MTP decode to avoid recurrent-state races.
- [`patches/operator.patch`](patches/operator.patch)
  - everything in `safe`;
  - row-per-warp CUDA `GATED_DELTA_NET`, forward-ported to the current
    recurrent rollback and fused-cache interface.

The safe variant produced byte-identical 512-token output versus the pinned
baseline. The operator variant changes floating-point reduction order, so its
text can differ while still passing CUDA-versus-CPU operator reference tests.

## Quick start

```bash
# Creates an isolated source checkout under .work/ and applies one patch.
./scripts/prepare-source.sh operator

# Run this on a CUDA build host with V100-class SM70 support.
./scripts/build-sm70.sh .work/llama.cpp-operator .work/build-operator

# Configure model paths without committing them.
cp config/qwen3.8-27b.env.example .env
$EDITOR .env

CONFIG=.env BUILD_DIR=.work/build-operator ./scripts/run-server.sh
```

Remote build from a workstation:

```bash
REMOTE=WZU_Server ./scripts/remote-build.sh operator
```

Benchmark the native llama.cpp endpoint:

```bash
python3 benchmarks/llama_native_bench.py \
  --url http://127.0.0.1:8000 \
  --prompt-file benchmarks/prompts/short.txt \
  --n-predict 512 --runs 3 \
  --output results/raw/operator-short.jsonl
```

## Optimization contract

Every change must include:

1. an upstream commit pin;
2. a baseline and candidate run with identical server arguments;
3. short and long-context measurements;
4. per-GPU memory data;
5. an operator reference test or deterministic output comparison;
6. a documented rollback path.

See [architecture](docs/architecture.md), [roadmap](docs/roadmap.md), and
[contributing](CONTRIBUTING.md) for the working model.

## Scope

In scope:

- SM70 CUDA kernels used during single-token decode;
- tensor-parallel overhead on exactly two GPUs;
- MTP/speculative decoding correctness and acceptance rate;
- KV-cache memory/performance trade-offs;
- long-context profiling and reproducible A/B automation.

Out of scope for now:

- high-concurrency serving throughput;
- training and fine-tuning;
- generic optimization claims without 2x V100 measurements.

## License and attribution

The orchestration and documentation in this repository are MIT licensed.
Patch code is derived from or forward-ports work proposed to `llama.cpp`; see
[`docs/attribution.md`](docs/attribution.md) and the upstream project license.
