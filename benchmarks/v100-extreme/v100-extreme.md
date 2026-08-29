# V100 Extreme Benchmark

This benchmark is the performance contract for Qwen3.8-27B Q8 on two Tesla V100 PCIe 32 GB GPUs. It targets single-user interactive inference, not concurrent serving.

## Fixed configuration

- Hardware: 2x Tesla V100 PCIe 32 GB on the same host
- Model: the production Qwen3.8-27B Q8 GGUF, unchanged
- Split: tensor split `1,1`
- Context: 262144 tokens
- KV cache: F16
- Parallel slots: 1
- Flash Attention: enabled
- MTP: enabled, `n-max = 3`
- Batch: 2048 unless the tested change explicitly tunes it
- Micro-batch: 512 unless the tested change explicitly tunes it
- Sampling: greedy, fixed seed
- Client: server loopback; WAN and Harness time are excluded

Changing weights, quantization, KV precision, context capacity, or prompt length invalidates a comparison.

## Extreme target

All throughput numbers are useful output tokens per second. A run passes only when the median and P10 across seven measured runs meet the target.

| Case | Current reference | Pass target | Improvement |
|---|---:|---:|---:|
| Cold prefill, 71351 tokens | 652.111 tok/s | 1000 tok/s | +53.4% |
| Decode, short context, 512 output tokens | 57.677 tok/s | 85 tok/s | +47.4% |
| Decode after 71351 tokens, 512 output tokens | 62.421 tok/s | 85 tok/s | +36.2% |
| Decode after 158000 tokens, 512 output tokens | To measure | 65 tok/s | - |
| Decode after 256000 tokens, 512 output tokens | To measure | 50 tok/s | - |
| Cached-prefix TTFT at 71351 tokens | To measure | 250 ms | - |

The primary V100-X score is:

```text
100 * cbrt((pp71k / 652.111) * (tg_short / 57.677) * (tg71k / 62.421))
```

The graduation requirement is `V100-X >= 145`, with every row above passing independently. A high score cannot hide a long-context regression.

## Measurement protocol

1. Record commit, build flags, CUDA driver, GPU clocks, temperature, power, CPU governor, NUMA placement, and command line.
2. Run two unmeasured warmups, followed by seven measured runs.
3. Clear server prompt cache before every cold-prefill run.
4. Generate 512 tokens for every decode case. Report TTFT separately from steady-state decode.
5. Record median, P10, P95, standard deviation, MTP accepted tokens, GPU utilization, memory-controller utilization, PCIe traffic, and peak VRAM per GPU.
6. Reject a run if either GPU throttles, another GPU process is active, or clocks differ from the recorded setting.
7. Compare every candidate against the reference build in alternating A/B order on the same host and within the same test window.

## Correctness gates

- All CUDA backend operator tests pass.
- The model completes all prompt lengths without context truncation or recurrent-state errors.
- Fixed-seed greedy output is checked against the reference. A changed reduction order may differ, but the first divergent logits must remain within the agreed numerical tolerance.
- Perplexity or task quality must not regress by more than 0.5% relative.
- A speed result obtained by lowering context, KV precision, model precision, or disabling work is rejected.

## Optimization milestones

| Milestone | pp71k | tg_short | tg71k | Purpose |
|---|---:|---:|---:|---|
| M1 | 750 | 65 | 68 | Profiling and launch/graph cleanup |
| M2 | 850 | 75 | 75 | Volta fused Q8 MMA and GDN work |
| M3 | 1000 | 85 | 85 | V100 Extreme graduation |

