# Attribution

The patch variants consolidate and forward-port work discussed upstream:

- `llama.cpp` PR [#26812](https://github.com/ggml-org/llama.cpp/pull/26812):
  split CUDA ARGMAX over multiple blocks for large rows.
- `llama.cpp` PR [#26827](https://github.com/ggml-org/llama.cpp/pull/26827):
  serialize multi-ubatch MTP decode execution.
- `llama.cpp` PR [#22587](https://github.com/ggml-org/llama.cpp/pull/22587):
  row-per-warp CUDA kernel for `GATED_DELTA_NET`.
- `llama.cpp` PR [#27173](https://github.com/ggml-org/llama.cpp/pull/27173):
  experimental chained MTP graphs and a reduced MTP proposal head. This lab
  reuses only the proposal-head idea in the faster stock single-step MTP path.

`operator.patch` forward-ports the GDN work to the recurrent-state rollback
and fused-cache interface at the pinned revision. The original upstream
project and patch authors retain their respective copyright. The upstream
`llama.cpp` project is distributed under the MIT License.
