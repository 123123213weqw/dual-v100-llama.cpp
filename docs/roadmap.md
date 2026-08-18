# Roadmap

## P0: reproducibility

- [x] Pin the validated upstream revision.
- [x] Publish safe and operator patch variants.
- [x] Publish short and 71K A/B summaries.
- [ ] Automate GPU clock, driver, CUDA, and topology capture.
- [ ] Add one-command baseline/safe/operator orchestration on separate ports.

## P1: correctness before speed

- [x] CUDA-versus-CPU ARGMAX reference coverage for 151,936 vocab rows.
- [x] CUDA-versus-CPU GDN coverage for K=1/2/3/4, KDA, permuted, and prefill.
- [x] Serialize MTP multi-ubatch recurrent-state mutation.
- [ ] Promote the reference cases into a standalone upstream-friendly test.
- [ ] Add long-running MTP acceptance and rollback stress tests.

## P2: single-user decode

- [x] Add and A/B an MTP-only proposal sub-vocabulary with full target verification.
- [ ] Profile kernel time by context length: 4K, 32K, 64K, 128K, 256K.
- [x] Tune GDN launch geometry for V100 SM occupancy and register pressure.
- [x] Add and validate two-query-head GQA K/V reuse for SM70 vector attention.
- [ ] Measure multi-block ARGMAX break-even points by vocab and row count.
- [ ] Investigate fused sampling without forcing a logits all-gather.
- [ ] Evaluate CUDA Graph capture for stable single-token decode shapes.

## P3: two-GPU communication

- [ ] Quantify PCIe/NVLink availability and all-reduce cost separately.
- [ ] Implement correct distributed top-k/argmax for split-axis-0 logits.
- [ ] Compare tensor split placement and peer-access settings.
- [ ] Attribute synchronization bubbles with Nsight Systems.

## P4: memory and long context

- [ ] Evaluate mixed KV types rather than all-F16 or all-Q8_0.
- [ ] Characterize KV bandwidth versus compute by context length.
- [x] Add a deterministic near-256K capacity prompt generator.
- [x] Execute a 255K-token single-V100 capacity/decode test with the production profile.
- [ ] Measure MTP acceptance rate as context grows.
