# 双 V100：保留精度的通信优化（2026-10-01）

## 本次采用的方案

仅增加 [通信环境变量覆盖层](../config/sm70-matched-allreduce.sh)，**不换模型、不换生产二进制、不改上下文、采样、输出预算和多模态投影器**。本报告独立于 8 月的 Q4 单卡与双卡实验，不能把不同基线的收益叠加。

生产上线状态以本文末尾的验收记录为准；Q8 packed/NUMA 不属于本次上线。

| 暖缓存上下文 | 原版均值 tok/s | 候选均值 tok/s | 配对几何平均收益 | 近似 95% 区间 |
|---|---:|---:|---:|---:|
| 短（99 tokens） | 54.684 | 55.922 | +2.11% | [-8.58, 14.05]% |
| 32768 | 50.046 | 52.051 | +4.01% | [3.10, 4.94]% |
| 131072 | 40.490 | 41.716 | +3.04% | [0.74, 5.39]% |
| 255000 | 37.318 | 38.771 | +3.88% | [2.91, 4.86]% |

每个上下文 3 个固定种子，各做原版 A／候选／原版 B；每次暖测均启动独立服务并恢复同一完整聊天模板 checkpoint，只重放最后 1 token，生成 1024 tokens。总计 36 次暖测及 12 次冷测，无截断，MTP 实际启用。原版取相邻 A/B 的几何平均；收益取配对比率的几何平均，并非表中两列算术均值之比。区间是 n=3 的近似配对 log-ratio 区间，未做多重比较校正。

**短上下文收益不可靠**：第三组原版漂移约 9.23%，区间跨零。32K／128K／255K 在本轮约提升 **4.01%／3.04%／3.88%**，不是所有任务都会提升这些比例。

### 冷启动代价

冷预填充每长度只有一组 A/C/B，不作稳定吞吐结论。短输入候选预填充约 1.846 秒，原版约 1.505 秒（约多 341 ms）；较长输入预填充变化约 -1.33% 至 +0.15%。这项优化针对暖 decode，不宣称提高冷启动速度。HTTP 首 token 总耗时保存在 JSON；没有另行测量流式 TTFT。

## 精度与功能证据

- 四个完整聊天上下文的固定长度输出及 MTP draft/accepted 计数均一致。
- 另做短／32K／128K／253K **独立全量预填充、自然 EOS**：12 个完整回答，包括 reasoning，全消息相同；JSON、前中后检索、算术与 gcd 检查通过。保留 8192-token 输出额度，253K + 8192 不超 262144。
- 三个图像 fixture × A/C/B：9 个独立输入自然 EOS 回答，包括 reasoning，全消息相同，12 个标签检索通过；保留 8192-token 额度。
- FP32 all-reduce 算子 blocks=3/8 共检查 17,663,488 个有限值，测试范围内与 CPU 参考逐位一致；不将此描述为所有 NaN/subnormal 或所有业务输入的普遍证明。

本次阈值为 **32767**。小消息走 FP32 自定义通信；大消息保持旧版 NCCL 冷 prefill 路径，保留其原有精度行为。`AR_BF16_THRESHOLD=0` / `AR_F16_WIRE=0` 不意味着整个模型所有运算都变成 FP32。曾测试 131072 阈值，改变冷 prefill 输出，因此拒绝。

## 运行环境与可复现边界

- 2× Tesla V100-PCIE-32GB，SM70，拓扑 NODE，无 NVLink。
- 驱动 580.178.04；容器 CUDA 12.8.1；NCCL 2.25.1。nvidia-smi 显示的 CUDA 13.0 是驱动支持上限，不是容器编译器版本。
- 既有功率上限 250 W；既有应用频率配置 graphics=1230 / memory=877 MHz，本轮未锁频、未修改 GPU 设置。**没有 Round7 全程时钟／显存采样，不能给出该轮精确峰值显存或排除频率漂移**；相邻双原版用于观察漂移。
- 原版与候选源码 HEAD 均为 `5a393c902ce5b0c08fb994a4feec8aec632d6019`；现有 `20260829-meta-tp-fixes-e5b1b578` release 保持不变。此次运行时覆盖层无需编译；不要把它当成对仓库所有上游 pin 的兼容声明。远程实验构建约束为 arch=70、GGML_CUDA=ON、GGML_NATIVE=OFF；本次未重建或宣称复原生产完整历史 CMake flags。
- llama-server SHA256：`e5b1b5782cd5b4acec5316ac5bd40123e0b0c9826643ae4148b0ec6e359f5fa2`。
- libggml-cuda SHA256：`205500226631ba3a45f3fbcb512b1e8d0957bbbfeeebe7ad74c0c6c3969f9d5d`。

原生产命令（容器内）保持如下；完整相关运行环境见 [结构化结果](../results/sm70-matched-allreduce-2026-10-01.json)：

```bash
/opt/llama/bin/llama-server -m /uncensored/Qwen3.8-27B-Uncensored-Q8_0.gguf --alias qwen3.8-27b-uncensored --host 0.0.0.0 --port 8000 --metrics -ngl all --split-mode tensor --tensor-split 1,1 --fit off -c 262144 --parallel 1 -b 4096 -ub 2048 -fa on -ctk f16 -ctv f16 --jinja --reasoning-format deepseek --mmproj /uncensored/Qwen3.8-27B-Uncensored-vision-f16.gguf --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling --backend-sampling --cache-prompt --cache-idle-slots -cram 131072 --slot-save-path /kv-cache
```

使用覆盖层前核对版本及哈希：

```bash
source config/sm70-matched-allreduce.sh
# 再用原有全部参数启动同一 release；不是重新构建。
```

对 Docker 必须把这 5 个变量显式传入新容器（`-e KEY=VALUE`），保留原容器全部其它配置。`source` 和 `docker restart` 不会修改一个现有容器的创建环境。不要直接套用旧的 `.env.example`，其 batch 等参数与本轮不同。

## 未采用的候选

- Q8 packed + NUMA 的筛选有弱收益，但独立 Round12 确认按用户要求中断，**未完成，不作为通过或生产效果**。
- Flash Attention 几何候选未显示可接受净收益，不上线。
- 本次不修改 CPU pinning，不叠加算子级 0.8%～1.4% 到模型 TPS。

## 生产验收与回滚

**已于 2026-10-01 17:10:20（CST）提交生产切换，部署单元终态 success，后续只读 audit 通过。** 8000/8001 健康，Docker health=healthy，重启策略仍为 `unless-stopped`。生产实际图像回答与已验收候选全消息一致；当前用户会话 API 重存后逐字节相同，洗数据进度哈希和原网关 PID 保持。原模型、CLI、Entrypoint、镜像、完整 release、F16 KV、262144 上下文、多模态及 CPU pinning 保持不变。

上线后的只读单次显存采样为 GPU0 27073 MiB / GPU1 25905 MiB，SM=1230 MHz，memory=877 MHz，功率上限各 250 W；不是基准全程峰值或显存节省结论。线上未重复跑大规模 TPS 压测，表中数字来自已完成的受控实验。

最初两次严格配置校验被 Docker 默认值表示差异拦截；第一次未暂停原服务，第二次自动恢复原服务并验证缓存逐字节一致。最终仅将 `OomKillDisable` 的 `null`/`false` 作等价比较（均不禁用 OOM killer），`true` 及所有其它字段仍严格检查；失败记录完整保留。手动回滚流程经过 CPU mocks，但未对已接入生产的新会话再做一次真实回滚演练。

切换必须保留原容器、原 release，保存当前真实会话 checkpoint 与进度；启动 cache manager 并等待其启动恢复结束，再最后恢复用户 checkpoint，API 重存并逐字节校验。图像 smoke 使用测试上下文，完成后恢复真实会话。回滚同样先保存**回滚当时**的新会话，不能使用上线前的旧快照覆盖之后的工作。

公开提交只含覆盖层与不带原始 prompt 的整理结果。不包含模型、二进制、会话 KV、生产 Docker inspect、凭据或原始任务资料。
