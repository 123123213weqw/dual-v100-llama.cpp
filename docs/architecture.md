# Architecture

## Repository model

```text
upstream.lock
      |
      v
clean llama.cpp checkout ---- patches/safe.patch
      |                       patches/operator.patch
      v
isolated SM70 build ---- correctness tests ---- A/B server
                                                  |
                                                  v
                              latency + tok/s + GPU telemetry
```

The repository does not vendor `llama.cpp`. `prepare-source.sh` checks out the
pinned upstream revision and applies exactly one complete variant patch.

## Reference serving topology

```text
client -> llama-server :8000
              |
              +-- tensor parallel split 1,1
                    |                 |
                 V100:0            V100:1
                   SM70              SM70
```

The primary target is one active sequence. Tensor parallelism is used to fit
the model and context across two 32 GiB V100s, not to maximize concurrent
request throughput.

## Hot paths

### ARGMAX

Large-vocabulary sampling can expose only a few very wide rows. A one-block-
per-row implementation leaves most SMs idle. The candidate performs a chunk
reduction across multiple blocks and then combines partial maxima.

### GATED_DELTA_NET

The row-per-warp kernel assigns four warps to a 16-row state tile while lanes
shard the Q/K dimension. It targets single-token recurrent decode and preserves
the current rollback/fused-cache state interface.

### MTP state ordering

Multi-ubatch speculative decode mutates recurrent state. Candidate execution
is serialized where concurrent ubatches could race on that state. This is a
correctness fix first; throughput is secondary.

## Variant strategy

- `baseline`: unmodified pinned upstream.
- `safe`: correctness fixes plus low-risk ARGMAX.
- `operator`: safe plus the tuned GDN kernel.

Production always retains all three artifacts so a kernel or model-specific
regression can be isolated without rebuilding.
