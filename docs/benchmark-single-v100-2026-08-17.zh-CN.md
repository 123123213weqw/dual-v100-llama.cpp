# 单张 V100 极限解码 A/B（2026-08-17）

## 目标与环境

- GPU：Tesla V100-PCIE-32GB，SM70，250 W，无 NVLink 依赖；
- 权重：Qwen3.8-27B `Q4_K_M`；
- 上下文容量：`262144`，单 slot；
- 服务：operator 补丁版 `llama.cpp`，CUDA 12.8；
- 通用参数：`-ngl all --split-mode none -b 2048 -ub 512 -fa on`；
- 测试方法：温度 0、固定 seed、单请求、无并发。

原始客户端输出保存在
[`results/raw/single-v100-2026-08-17`](../results/raw/single-v100-2026-08-17)，
精选数据见
[`results/single-v100-summary-2026-08-17.tsv`](../results/single-v100-summary-2026-08-17.tsv)。
MTP proposal 子词表的直接 A/B 保存在
[`results/single-v100-subvocab-summary-2026-08-17.tsv`](../results/single-v100-subvocab-summary-2026-08-17.tsv)，
原始输出位于 `results/raw/single-v100-subvocab-2026-08-17/`。

## 结论

最终吞吐配置是：

```text
Q4_K_M weights
Q4_0 K/V cache
MTP draft max = 1
draft backend sampling = enabled (upstream single-GPU default)
main backend sampling = disabled
batch = 2048, ubatch = 512
context = 262144, parallel = 1
multimodal projector = enabled
V100 application clocks = 877,1380 MHz
MTP proposal vocab = 131072 (target verification remains full vocab)
```

增加 MTP proposal 子词表后，最终 512-token 短上下文三次均值为
**44.201 tok/s**，范围 `44.107–44.330 tok/s`；相对前一生产配置的
`43.420 tok/s` 提升 **1.80%**，512-token 输出 SHA-256 与 control
完全相同。同一配置已经通过文本和
64×64 PNG 图片的 OpenAI-compatible `/v1/chat/completions` 冒烟测试。

### MTP proposal 子词表

`mtp-subvocab.patch` 只裁剪 MTP 草稿头；target 模型仍用完整词表验证每个
候选 token，因此它不改变 target 的采样空间。该路径只在 draft backend
sampling 已挂载时启用，避免 raw-logits 路径按完整 `n_vocab` 读取越界。

四类 chat 工作负载（中文、C++、数学推导、中英混合 CUDA）各生成 256
tokens 的直接 A/B 如下：

| proposal 词表 | 四任务 decode 均值 | 相对 control | 接受率/输出 |
|---:|---:|---:|---|
| 完整词表（control） | 44.977 tok/s | - | 基准 |
| 196608 | 45.229 tok/s | +0.56% | 四任务不变 |
| **131072** | **45.576 tok/s** | **+1.33%** | 四任务不变，输出 hash 全相同 |
| 114688 | 45.493 tok/s | +1.15% | 中文接受率从 84.78% 降至 79.58% |

因此选 `131072`，而不是追求更小词表。短 completion 两次均值从
`42.762` 提升到 `43.493 tok/s`（+1.71%）；64,810-token prompt 的同机
隔离 A/B 从 `27.530` 提升到 `27.802 tok/s`（+0.99%），输出 hash 与
MTP 接受率均相同。

### 255K 实际容量验证

使用 `benchmarks/generate_repeated_prompt.py --repeats 255000` 本地生成
确定性 fixture；对当前 Qwen tokenizer，`" test"` 恰为一个 token，因此
实测输入正好是 `255000` tokens，而不是只验证 262K KV 预分配：

| prompt | generation | prefill | decode | MTP 接受率 | 峰值显存 |
|---:|---:|---:|---:|---:|---:|
| 255000 | 128 | 224.352 tok/s | **12.617 tok/s** | 100.00% | 25145 MiB |

该 fixture 是低熵容量测试，100% MTP 接受率和 decode 数字不能代表普通
对话；真实性能仍以 64,810-token 长文本 A/B 为准。它证明单张 V100 在
保留 `262144` context、Q4_0 KV、mmproj 和 MTP 的同时，确实能处理接近
完整窗口的输入并继续生成。

同时实测了上游 PR #27173 的 chain MTP。该分支在 V100 上即使 `n-max=1`
也只有 `38.550 tok/s`，低于当前 control 的 `42.742 tok/s`；`n-max=3`
和 4 又分别降到 `33.481`、`30.117 tok/s`。因此没有整体合入 chain，
只把对单步 MTP 有收益的 proposal-head 裁剪移植到 pinned 快路径。

## 关键 A/B

### MTP 长度

| MTP 最大草稿长度 | 短上下文 decode |
|---:|---:|
| 0（关闭） | 30.570 tok/s（Q8 KV） |
| 1 | **41.926 tok/s** |
| 2 | 40.380 tok/s |
| 3 | 38.860 tok/s |
| 4 | 34.480 tok/s |
| 5 | 30.269 tok/s |

该工作负载的 MTP 接受率不足以抵消长草稿的额外计算，`n-max=1` 明显优于
原生产值 3。关闭 draft backend sampling 后，MTP3 从 `38.860` 降到
`37.035 tok/s`，所以单卡必须保留后端草稿采样。

### KV cache

在 MTP1、384-token 搜索测试中：

| KV 类型 | decode | 说明 |
|---|---:|---|
| Q8_0 | 41.926 tok/s | 较高精度档 |
| Q5_0 | 35.285 tok/s | SM70 上对应量化核不划算 |
| Q4_0 | **42.711 tok/s** | 吞吐与显存最佳 |

Q4 KV 是吞吐优先档，会改变低位精度和生成轨迹；需要更保守的质量档时切回
Q8_0。它不是无损优化。

### 64K 实际上下文

输入为 `64810` tokens，均启用 mmproj 与 MTP1：

| KV 类型 | decode | prefill | 峰值显存 |
|---|---:|---:|---:|
| Q8_0 | 20.237 tok/s | 468.804 tok/s | 29239 MiB |
| Q4_0 | **21.847 tok/s** | **471.947 tok/s** | **25143 MiB** |

Q4 KV 在真实 64K decode 上提升 `7.96%`，并节省 `4096 MiB`。模型在完整
`262144` context 预分配下成功装载，因此 256K 容量成立。

### 补丁与时钟

- 最终 Q4/MTP1 条件下，operator 为 `42.711 tok/s`，safe 为
  `40.976 tok/s`，operator 的行/warp GDN 内核提升 `4.24%`；
- 将 GPU0 application clock 从默认 `1230` 固定到 `1380 MHz` 后，
  512-token 三次均值从 `41.449` 提高到 `43.420 tok/s`，且方差显著下降；
- Q4/MTP1 相对 Q4/no-spec 的短上下文提升为 `34.96%`。

## 已部署实例

容器名为 `qwen38-27b`，只占用 GPU0，绑定主机
`127.0.0.1:8000`，restart policy 为 `unless-stopped`。GPU1 保持空闲。
服务启动后显存约 `24757 MiB`，保留约 8 GiB 余量。

复现当前容器（脚本默认挂载 `operator-subvocab-20260817/bin` 并注入
`LLAMA_MTP_SUBVOCAB=131072`）：

```bash
./scripts/deploy-single-v100-docker.sh
```

脚本会先固定 GPU0 application clock；该时钟设置在服务器重启后需要重新
执行。视觉侧显式设置 `--image-min-tokens 1024`，避免 Qwen-VL 对目标定位类
任务的低分辨率警告。

回滚 proposal 优化可设置 `MTP_SUBVOCAB=0`；回滚质量档只需把
`-ctk/-ctv` 改回 `q8_0`；完整回滚补丁则把二进制挂载改回
`operator-ab-20260816/optimized/bin`。
