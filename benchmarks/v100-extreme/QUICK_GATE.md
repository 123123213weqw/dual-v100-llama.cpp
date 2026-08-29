# V100 quick gate

`quick_gate.py` is a **single-pass gate**, not the seven-run V100-X benchmark.
It is designed to qualify a specific `llama-server` binary/config without
controlling production.

For WZU, do not use direct managed mode: the host lacks CUDA runtime, cuBLAS,
and NCCL shared libraries. `maintenance_ab.py` starts an audited CUDA container
and supplies `external_server.base_url` plus a followed `server_log`; in that
mode the quick gate owns and signals no server process.

## Safety contract

- The script contains no Docker/systemd/SSH production-control path.
- It rejects port `8000`, binds only to loopback, and rejects an occupied port.
- Before launching, it requires `nvidia-smi` to report **zero CUDA compute
  clients**. If production or any other GPU job is alive, it fails closed. It
  checks again after model load and refuses to send a request if any CUDA PID
  other than its owned candidate appeared during the startup window.
- It starts the candidate in a new process group and signals only that exact
  owned process group during cleanup.
- It never changes clocks, persistence mode, power limits, models, or caches of
  another service.

Production must therefore be drained/stopped by an operator outside this
script. The gate itself will never do that work.

## Workloads and gates

1. `POST /completion`: exactly 71,351 input token IDs and 512 greedy output
   tokens (`cache_prompt:false`). Content and raw token-ID hashes are recorded.
2. `POST /v1/chat/completions`: the exact streaming/thinking/tool wire shape
   used by DeepSeek Harness, with a nested JSON grammar and strict validation.
3. A second Harness-shaped request replays the first assistant
   `reasoning_content`, `content:""`, `tool_calls`, and tool result, exercising
   cached-prefix continuation and cross-request MTP state.

For every case the bundle contains request JSON, raw response/SSE, canonical
response JSON, Prometheus snapshots and deltas, TTFT, decode/prefill throughput,
MTP acceptance/accepted length, and the corresponding server-log slice. The
gate fails on either of these server markers:

- `inconsistent sequence positions`
- `llama_decode[...] returned -1` (including deferred decode forms)

## Use on WZU

Use `maintenance_ab.py`; it creates the external-client config, follows Docker
logs, and guarantees ingress/production restoration. Do not run the direct
example on WZU.

## Direct mode on a CUDA-equipped host

Sync the bench directory with the source tree, copy the example config, and
edit only the candidate binary/model/config paths as required:

```bash
cd ~/codex-build/llama.cpp-operator-opt
cp benches/v100-extreme/quick-gate.example.json /tmp/v100-gate.json
${EDITOR:-vi} /tmp/v100-gate.json

# Safe validation only: does not query GPUs and does not start the server.
python3 benches/v100-extreme/quick_gate.py \
  --config /tmp/v100-gate.json --print-plan

# Run only after the operator has independently drained any production service.
python3 benches/v100-extreme/quick_gate.py \
  --config /tmp/v100-gate.json
```

The last stdout line is the immutable result directory, under
`benches/v100-extreme/results/` by default. `summary.json` is machine-readable;
`summary.md` is the compact report. Exit status is `0` for pass, `1` for a test
failure, and `2` for a fail-closed preflight error.

To qualify a different binary without rewriting the config:

```bash
python3 benches/v100-extreme/quick_gate.py \
  --config /tmp/v100-gate.json \
  --binary /absolute/path/to/candidate/bin/llama-server
```

Set `expected_legacy_content_sha256` after one accepted reference run to turn
the legacy content digest into a hard A/B correctness gate. The token-ID digest
is always emitted independently because decoded text alone can hide tokenization
differences.
