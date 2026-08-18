# 单张 V100 SM70 内核调优（2026-08-18）

本轮只优化单用户、单 token decode，不改变 `262144` context、Q4_0 KV、
MTP1、131072 proposal vocabulary、mmproj 或模型量化。生产容器在每组测试后
自动恢复，候选二进制始终位于独立的版本化目录。

## 方法

- GPU0 固定 application clock 为 `1380/877 MHz`，GPU1 的训练任务不改动；
- 每个 64K case 先完成一次 64810-token prefill/warmup，再等待 30 秒；
- 记录 4 次 decode，轮次间隔 15 秒，prompt cache 命中后每轮只处理 4 个
  prompt tokens；
- 逐秒记录 SM clock、温度、功耗和 power-cap throttle；严格 A/B 中 decode
  全程为 1380 MHz，未出现 power-cap throttle；
- 所有 case 使用相同 seed、temperature、MTP 和服务参数。

## Nsight 结论

`flash_attn_ext_vec<D=256,Q4_0,Q4_0>` 在 SM70 上使用 128 threads/block，
每线程 250 个寄存器，单 head decode 的实际 occupancy 只有 6.25%。模型有
24 个 query heads、4 个 KV heads，即 GQA ratio 为 6。原内核按 query head
启动 block，同一 KV head 的 K/V 会被六组 block 重复处理。

先后验证了三类低风险 launch 变体：

| 变体 | 64K decode | 相对同协议基线 | 结论 |
|---|---:|---:|---|
| FA 256 threads | 27.603 | -0.28% | block 变大但有效并行度不增 |
| GDN 8 warps × 2 rows | 27.677 | -0.01% | 寄存器下降但端到端持平 |
| FA min-blocks=3 | 27.601 | -0.29% | 寄存器 250→168，但 spill 抵消 occupancy 收益 |

这些结果说明仅修改 launch geometry 已到边际，必须减少重复工作。

## GQA head grouping

新 fast path 用 `GGML_CUDA_FATTN_VEC_GQA_HEADS=2` 启用。单 token 且
GQA ratio ≥ 2 时，一个 block 同时计算同一 KV head 下的两个 query heads：

1. 两个 Q 各自保持独立的 QK dot、softmax 和输出累加器；
2. 每个解量化后的 V row 被两个 attention weight 复用；QK dot 仍独立，
   但相邻访问在同一 block 内获得更好的 K cache locality；
3. mask、sink、ALiBi slope 和输出索引仍按 query head 独立处理；
4. prefill 和非 GQA 模型保持原路径；默认值为 1，可在编译时完全回滚。

严格相邻 A/B：

| 变体 | 短上下文 | 64K decode | 64K 标准差 | 峰值显存 |
|---|---:|---:|---:|---:|
| control | 41.288 | 28.017 | 0.017 | 25143 MiB |
| GQA×2 | 41.143 | **28.153** | 0.018 | 25143 MiB |
| 变化 | -0.35% | **+0.49%** | - | 0 MiB |

短上下文 KV 很小，双 head 的寄存器和调度开销略高；64K 开始，共享 K/V
读取与解量化产生稳定净收益。第二组 GQA×2 64K 四次结果为
`28.177/28.159/28.136/28.143 tok/s`。

## 255K 全上下文门槛

最终容量测试使用实际 `255000`-token prompt、`262144` context 和 128-token
输出。每个二进制先完整 prefill 一次，随后使用 prompt cache 记录两轮 decode：

| 变体 | 255K decode | 标准差 | 峰值显存 | MTP acceptance |
|---|---:|---:|---:|---:|
| control | 12.617 | - | 25145 MiB | 1.00000 |
| GQA×2 | **13.001** | 0.014 | 25145 MiB | 1.00000 |
| 变化 | **+3.04%** | - | 0 MiB | 0 |

候选两轮分别为 `13.0108/12.9906 tok/s`。两边 128-token 输出 SHA-256
均为 `5726c1347b8a76cd4f4b84456c94247e9d0d17e369a084ca885b8391e084f55f`，
MTP 都接受 `63/63` proposals、mean len `2.00`。收益随上下文增长而扩大，
与该 fast path 减少长 KV 扫描中的重复流量这一设计目标一致。

## 正确性

- `test-backend-ops -o FLASH_ATTN_EXT -b CUDA0` 通过，包括 D=256、Q4_0、
  GQA、mask、sink 和 ALiBi 组合；
- GDN 8×2 候选的 36/36 CUDA-versus-CPU reference cases 通过；
- short 与 64K 的 control/candidate 输出 SHA-256 分别完全一致；
- 64K MTP acceptance 同为 `0.92308`（108/117），mean len 同为 `1.92`。
- 255K 输出 SHA-256、MTP acceptance 和峰值显存也完全一致。

原始汇总见
[`results/sm70-tuning-summary-2026-08-18.tsv`](../results/sm70-tuning-summary-2026-08-18.tsv)。
255K 原始输出见
[`fa128-gqa2-gdn4x4-255k.255k.out`](../results/raw/sm70-tuning-2026-08-18/fa128-gqa2-gdn4x4-255k.255k.out)。

## 部署与回滚

通过 255K 门槛后，胜出构建以独立版本目录部署：

`/home/wzu/models/Qwen3.8-27B-GGUF/operator-subvocab-gqa2-20260818/bin`

旧构建没有覆盖，回滚时设置：

```bash
BIN_ROOT=/home/wzu/models/Qwen3.8-27B-GGUF/operator-subvocab-20260817/bin \
  ./scripts/deploy-single-v100-docker.sh
```
