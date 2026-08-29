# 双 V100 当前生产配置（2026-08-29）

## 硬件与模型

- GPU：2 × NVIDIA Tesla V100 32 GiB，SM70
- 目标模型：Qwen3.8-27B-Uncensored `Q8_0`
- 多模态投影：Qwen3.8-27B-Uncensored vision `F16`
- 工作负载：单用户、长上下文、OpenAI 兼容服务
- 最大上下文：262,144
- Tensor Parallel：`--split-mode tensor --tensor-split 1,1`
- KV：K/V 均为 F16
- MTP：目标模型自带 MTP head，draft max = 3

## 服务器参数

```text
-ngl all
--split-mode tensor
--tensor-split 1,1
--fit off
-c 262144
--parallel 1
-b 4096
-ub 2048
-fa on
-ctk f16
-ctv f16
--jinja
--reasoning-format deepseek
--spec-type draft-mtp
--spec-draft-n-max 3
--spec-draft-backend-sampling
--backend-sampling
--cache-prompt
--cache-idle-slots
-cram 65536
--slot-save-path /kv-cache
```

运行时优化开关：

```text
GGML_META_ASYNC_HOST_COPY=1
GGML_META_PARALLEL_LAUNCH=1
LLAMA_EXPERIMENTAL_TP_BACKEND_SAMPLING=1
LLAMA_EXPERIMENTAL_TP_LOCAL_TOPK=1
LLAMA_EXPERIMENTAL_TP_LOCAL_TOPK_K=10
LLAMA_EXPERIMENTAL_MTP_MIRROR_LAYER=0
LLAMA_MTP_DEFER_VERIFY=1
LLAMA_MTP_DEFER_VERIFY_MIN_POS=32768
LLAMA_MTP_CANDIDATE_K=10
LLAMA_MTP_ALIGN_TARGET_SAMPLER=1
```

完整容器启动与无中断回滚流程见
[`scripts/run-qwen38-27b-production.sh`](../scripts/run-qwen38-27b-production.sh)。

## 冷 KV 持久化

[`scripts/kv-cache-manager.py`](../scripts/kv-cache-manager.py) 在请求结束并安静
30 秒后保存 slot checkpoint。它使用 `latest.next.bin -> latest.bin` 原子轮换，
保留 `previous.bin`，并在 llama-server 重启后恢复最后一个完整 checkpoint。

对应 systemd unit：
[`systemd/qwen38-kv-manager.service`](../systemd/qwen38-kv-manager.service)。

默认策略：

- poll：1 秒
- quiet window：30 秒
- 最短保存间隔：300 秒
- 最小 token 增量：8192
- 部署前必须确认 slot 不在处理请求
- 新容器健康检查失败时自动恢复旧容器

## MTP 与 DFlash2 Q8 公平 A/B

同一安全审查提示、512 输出 token。长上下文用确定性合成前缀，实际
`prompt_n = 116,934`。

| 条件 | MTP prefill | DFlash2 prefill | MTP decode | DFlash2 decode |
|---|---:|---:|---:|---:|
| 94-token prompt | 47.76 tok/s | 56.03 tok/s | **53.88 tok/s** | 45.89 tok/s |
| 116,934-token prompt | **833.37 tok/s** | 607.82 tok/s | **43.34 tok/s** | 33.00 tok/s |

长上下文端到端：

- MTP：153.03 秒
- DFlash2 Q8：208.57 秒

接受率：

| 条件 | MTP | DFlash2 |
|---|---:|---:|
| short | 298 / 635 = 46.9% | 354 / 1096 = 32.3% |
| 116,934 tokens | 298 / 638 = 46.7% | 335 / 1227 = 27.3% |

DFlash2 drafter 针对官方 Qwen3.8-27B 训练，与当前 Uncensored 目标权重不完全
匹配；同时双卡 Tensor Split 需要未合并修复，NCCL 路径仍不稳定。因此生产继续
使用 MTP。原始机器可读结果见
[`results/2026-08-29/mtp-vs-dflash-summary.json`](../results/2026-08-29/mtp-vs-dflash-summary.json)。

## 补丁与复现

当前源码差异保存在
[`patches/production-current-2026-08-29.patch`](../patches/production-current-2026-08-29.patch)，
基线为 `upstream.lock` 中的 `ad1de39e0708e3ced9c71bb3c82d93a2c046a73f`。

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout ad1de39e0708e3ced9c71bb3c82d93a2c046a73f
git apply /path/to/production-current-2026-08-29.patch
```

仓库不会上传模型、KV checkpoint、API key 或本机环境文件。
