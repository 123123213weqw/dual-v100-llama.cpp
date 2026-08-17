# Qwen3.8-27B / 2×V100 算子优化 A/B 报告

日期：2026-08-16
上游基线：`ad1de39e0708e3ced9c71bb3c82d93a2c046a73f`
生产旧版：llama.cpp b10423 / `a94d563ed`
统一参数：tensor parallel `1,1`、context 262144、parallel 1、batch 2048、ubatch 512、FA on、MTP n-max 3、draft backend sampling off。

## 实现

- PR #26812：CUDA 多 block ARGMAX。
- PR #26827：MTP 多 ubatch decode 串行化，修复 recurrent-state race。
- PR #22587：row-per-warp GATED_DELTA_NET，已 forward-port 到当前 recurrent rollback/fused-cache 接口。
- TP backend sampling：实现了实验开关并做了真实启动测试；Meta backend 在 split-axis-0 logits 的 per-row sampler 上触发断言，不能安全上线，最终生产未启用。
- tensor parallel + q8_0 KV：当前上游已可运行，完成短/71K 测试。

## A/B 结果

| Case | 短上下文 decode | 71K decode | 71K prefill | 单卡显存 |
|---|---:|---:|---:|---:|
| master base + f16 KV | 57.400 | 61.747 | 655.332 | 23735 MiB |
| safe(ARGMAX+MTP fix) + f16 KV | 57.303 | — | — | 23665 MiB |
| operator(GDN+ARGMAX+MTP fix) + f16 KV | **57.677** | **62.421** | 652.111 | 23735 MiB |
| master base + q8_0 KV | 55.836 | 57.355 | 655.350 | 20879 MiB |
| operator + q8_0 KV | 56.645 | 55.282 | 643.497 | 20879 MiB |

f16 operator 相对 master base：短上下文 `+0.48%`，71K decode `+1.09%`。
相对旧生产 b10423：短上下文 `+0.56%`，71K decode `+0.80%`。
q8_0 相对 f16 节约 `2856 MiB/GPU`，但 71K decode 慢 `11.44%`，因此不用于生产。

## 正确性与失败项

- GATED_DELTA_NET：CUDA0 对 CPU reference，`36/36` 通过，覆盖 K=1/2/3/4、KDA、permuted、prefill。
- ARGMAX：CUDA0 对 CPU reference，`10/10` 通过，覆盖 151936 vocab 和多行输入。
- safe variant 与 master base 的 512-token 输出 SHA256 完全一致。
- row-per-warp GDN 改变浮点归约顺序，因此 operator variant 的生成文本可不同，但算子 reference 容差测试全部通过。
- TP backend sampling 失败点：`ggml-backend-meta.cpp:543`, `src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0`；需要分布式 top-k/softmax/argmax 或 logits all-gather，不是删除 guard 就能正确工作。

## 上线结论

生产选择 `operator + f16 KV`；保留 `safe`、`base` 和旧官方镜像三层回滚。服务端口、模型名和 Harness 配置均不变。
