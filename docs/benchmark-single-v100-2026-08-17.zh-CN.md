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
```

最终 512-token 短上下文三次均值为 **43.420 tok/s**，范围
`43.356–43.459 tok/s`，峰值显存 `24771 MiB`。同一配置已经通过文本和
64×64 PNG 图片的 OpenAI-compatible `/v1/chat/completions` 冒烟测试。

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

复现当前容器：

```bash
./scripts/deploy-single-v100-docker.sh
```

脚本会先固定 GPU0 application clock；该时钟设置在服务器重启后需要重新
执行。视觉侧显式设置 `--image-min-tokens 1024`，避免 Qwen-VL 对目标定位类
任务的低分辨率警告。

回滚质量档只需把 `-ctk/-ctv` 改回 `q8_0`；回滚补丁则把二进制挂载从
`optimized/bin` 改为 `safe/bin`。
