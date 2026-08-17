# Contributing

## Change types

- `kernel`: SM70 CUDA kernel implementation or launch geometry.
- `runtime`: scheduler, tensor-parallel, MTP, or state-management behavior.
- `memory`: KV cache, scratch allocation, or model placement.
- `bench`: measurement, telemetry, and correctness tooling.

## Required evidence

Open a focused change with:

- exact `llama.cpp` base and candidate commit IDs;
- complete build flags and server command;
- GPU model, driver, CUDA runtime, clocks, and power limit;
- warm-up policy and at least three measured runs;
- prompt-token count, generated-token count, prefill rate, decode rate, TTFT,
  and memory per GPU;
- correctness evidence against CPU reference or an unmodified baseline;
- results for both a short prompt and at least one 64K+ context prompt.

Do not combine unrelated kernels in one benchmark claim. If a patch changes
floating-point reduction order, explicitly distinguish numerical tolerance
from byte-identical generation.

## Workflow

```bash
./scripts/check-patches.sh
REMOTE=WZU_Server ./scripts/remote-build.sh operator
```

Run the candidate on a different port from production and store machine-
readable output under `results/raw/`. Only curated, prompt-free summaries are
committed.
