# WZU dual-V100 maintenance A/B

## Read-only deployment audit (2026-08-20 03:29–03:32 UTC)

### Production container

- Container: `qwen38-27b`, ID prefix `ccc2170d2a46`, running and healthy.
- Immutable CUDA image ID:
  `sha256:07247b31345790c74ee47631ce792dd1183a4e83ec7bccec240c1708e53b3a1f`
  (`nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`).
- Host binding: only `127.0.0.1:8000 -> container:8000`.
- Restart policy: `unless-stopped`; IPC is host; all GPUs requested;
  `CAP_SYS_NICE` is present.
- Release bind:
  `/home/wzu/v100-x1/releases/20260820-operator-ab-131b7b57/bin:/opt/llama/bin:ro`.
- Model binds are read-only at `/official` and `/uncensored`.
- Current release manifest SHA-256:
  `e88e1f8294495bf35eecfca15c404c77ce4cac3e06aa1c0759401a9a241351f7`.
- `/health` returns 200/`ok`; `/slots` returns 200. The only slot was idle
  (`is_processing:false`) while retaining a 110,855-token prompt state.
- `/metrics` returns **501** because production was not started with
  `--metrics`. Candidate containers add it explicitly.

### GPU and storage

- 2× Tesla V100 PCIe 32 GiB, driver `580.173.02`.
- Both GPUs are owned only by production PID `978202`; observed allocations
  were about 27.2/26.1 GiB.
- GPU-to-GPU topology is `NODE`—same NUMA node, but no NVLink.
- Root filesystem: 1.8 TiB total, 1006 GiB free; inode use 7%.

### Public ingress

```text
https://api.47.245.37.106.nip.io
  -> Tokyo HAProxy TLS SNI mux :443
  -> Tokyo Caddy :8444 (exact Bearer header gate)
  -> Tokyo 127.0.0.1:18000
  -> qwen-tokyo-tunnel.service reverse SSH
  -> WZU 127.0.0.1:8000
```

`caddy`, `haproxy`, and `qwen-tokyo-tunnel.service` were active. The public
endpoint returned 401 without a key and the tunnel-local health endpoint
returned 200. No secret value was read or recorded.

## Why the candidate must use Docker

The host release directory contains the llama/ggml shared objects, but a host
`ldd` with the release directory in `LD_LIBRARY_PATH` still reports these as
missing:

```text
libcudart.so.12
libcublas.so.12
libnccl.so.2
```

They exist in the audited CUDA image, not on the WZU host. Therefore the old
direct-process quick-gate example cannot qualify this deployment. The
maintenance runner starts each candidate inside the immutable CUDA image and
runs `quick_gate.py` in **external client-only mode** on the host.

Do not use `/home/wzu/bin/qwen-model` for an A/B. It calls `docker rm -f`, has
old b2048/u512 flags, and treats the current engine label
`v100-x1-operator-ab` as unknown; its fallback can restore the wrong release.

## Safe runner

Files:

- `maintenance_ab.py` — local orchestrator; offline dry-run by default.
- `maintenance-ab.example.json` — audited topology and exact reference config.
- `quick_gate.py` — request/metrics/log/correctness gate; now supports an
  externally managed loopback candidate.

Safe dry-run (no SSH connection):

```bash
python3 benches/v100-extreme/maintenance_ab.py \
  --config benches/v100-extreme/maintenance-ab.example.json
```

Before execution, replace the B release path and pin its manifest hash. Actual
execution additionally requires the explicit confirmation phrase:

```bash
python3 benches/v100-extreme/maintenance_ab.py \
  --config /absolute/path/to/filled-maintenance-ab.json \
  --execute --confirm MAINTENANCE-AB-WZU-V100
```

No execution was performed while creating or validating these files.

## State machine and rollback

1. Hold `flock` sessions on WZU and Tokyo.
2. Snapshot container ID/config fingerprint, slot state, GPU ownership and the
   byte-exact Caddyfile.
3. Validate a temporary Caddy config, arm a timed Tokyo rollback watchdog, and
   gracefully reload only the marked Qwen site to return 503. Existing streams
   remain on the old config; new work is rejected.
4. Require three consecutive idle `/slots` polls.
5. Arm a WZU watchdog that removes only run-ID-owned candidates and starts the
   same production container after the timeout.
6. Stop—but never remove—production and require zero CUDA processes.
7. Run A then B in disposable, uniquely named CUDA containers on
   `127.0.0.1:18081`; collect the quick-gate result for each.
8. Start the original container ID and require Docker health, HTTP health,
   exclusive GPU ownership, and an unchanged config fingerprint.
9. Only then restore and validate the byte-exact Caddyfile, verify public 401
   and tunnel-local 200, cancel watchdogs, and release locks.

If the local orchestrator dies, the WZU watchdog first restores production;
the Tokyo watchdog waits until tunnel health succeeds before reopening ingress.

## Exact candidate launch delta

The complete shell-escaped command is printed by the dry-run. Relative to the
audited production command, each candidate changes only:

- unique container name and maintenance labels;
- `--restart=no`;
- host port `18081` instead of production port `8000`;
- release bind selected by A or B;
- adds `--metrics` for counter deltas;
- shorter 10-second container health interval.

The model, mmproj, image ID, dual-GPU tensor split, 262,144 context, F16 KV,
b4096/u2048, Flash Attention, MTP n-max 3, backend sampling, allreduce/NCCL and
TP-local-topK environment are otherwise identical.

For an intentional environment-only tuning A/B, a variant may contain an
`env` object. Those keys override the shared `candidate_runtime.env` only for
that disposable candidate and are recorded in the rendered command and result
config. Omit the object for binary-only qualification.

## Remaining risks

- Maintenance is real downtime: one 27B model cannot coexist with production
  in 2×32 GiB VRAM.
- Caddy graceful reload is required to make drain strict. Stopping the reverse
  SSH tunnel instead would sever in-flight streams and is not used.
- A hard host/Tokyo power loss can outlive in-memory SSH locks. The timed
  watchdogs cover orchestrator loss, not complete machine loss.
- A manual Caddy edit during maintenance causes restoration to fail closed
  rather than overwrite an operator's change; ingress remains in maintenance.
- The B variant is intentionally a placeholder and execution is rejected until
  its absolute release path and manifest SHA-256 are pinned.
