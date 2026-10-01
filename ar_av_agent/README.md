# Verified-Prefix AR Audio/Video Agent · v0

这是 `vedio-agent` 的新增研究原型，不修改仓库原有的稀疏候选并发实验。主机制固定为 **生成下一个候选 → 检查 → 接受并提交 / 丢弃候选重试**。已经提交的视频和音频不回退，不学习回退深度，不把事后配音当作原生音视频生成。

**交付边界：这是可运行的控制与评测准备代码，不是已经完成的音视频模型产品，也没有论文实验分数。** CPU 状态机、数据解析和经验准入测试已经执行；LongLive 参考适配器按官方源码接口编写，尚未用真实 checkpoint/GPU 验证；通用场景的原生联合 AR 音视频模型仍需完成具体驱动接入。不能把 mock 的通过率当作任务成功率。

## 1. 现在包含什么

| 部分 | 内容与状态 |
|---|---|
| 内循环 | 已实现：不可变前缀、两模态共同验收、失败后缓存恢复、局部重试/改 prompt、预算停止 |
| 提示词 | 可直接读取：Planner、Verifier、Controller、Evolver、独立最终评审，另有明确非官方的 CineScope 分析提示 |
| 模型接入 | LongLive1.0 纯视频参考适配器；`local_module:factory` 原生 AV 驱动接口；CPU mock |
| 评测准备 | StoryEval / TempAct / AVGen 官方数据下载、离线导入、来源锁定、输出映射；官方评分依赖仍按上游安装 |
| 经验进化 | TRAIN 轨迹提出文本规则，独立 DEV 配对验证后保存新策略；TEST 不写回 |
| 记录 | 原始任务、固定计划、随机种子、每次候选、拒绝原因、成本、前缀文件哈希；正式分数默认为 null |

`accepted` 只表示通过当前 verifier，而不是数学上保证语义正确。这个区别不改变固定前缀机制；最终仍由独立评测检验。连续失败就返回未完成，不偷偷放宽标准或接受最不差的失败候选。

## 2. 无 GPU / 无 API 的自检

需要 Python 3.10+；核心包无第三方依赖，测试需要 pytest。

```bash
cd ar_av_agent
python -m pytest -q
python -m arav.cli demo --out runs/smoke-001
python -m arav.cli bench-list
```

Demo 用脚本化状态机和静音 WAV 检查事务，不生成真实视频。它有意先失败一次，再验证两路缓存都恢复且只提交后续候选。`task_success`、`official_benchmark_score` 保持 null。

也可执行 `pip install -e '.[test]'` 安装 `arav` 命令。模型依赖单独使用上游环境，不用一个大 requirements 文件混合互相冲突的视频、音频和评价环境。

## 3. 准备真实 benchmark

以下命令只下载公开文本、保留许可/说明、生成统一任务列表和 `manifest.json`，不会自动启动付费模型或 GPU 评测。首次运行需要机器可以访问 GitHub；本次交付未在当前无联网容器中下载完整官方数据。

```bash
python -m arav.cli bench-prepare storyeval --out data/storyeval
python -m arav.cli bench-prepare tempact --out data/tempact
python -m arav.cli bench-prepare avgen --out data/avgen
```

每次先解析一次上游 commit，再按固定 commit 下载全部文件并记录 SHA256；StoryEval/TempAct 还核对本次审查的 Git blob 哈希。上游变动时默认停止，审查后才用 `--allow-update`。获取失败的 manifest 保持 `complete=false`。

已经下载数据时，可以离线导入，例如：

```bash
python -m arav.cli bench-import storyeval /path/to/StoryEval/prompts/all_prompts.json --out data/storyeval.tasks.jsonl
```

StoryEval 保留原始 `event_list` 与官方视频文件名；TempAct 保留 CSV 来源及 category，不根据文件名猜难度；AVGen 保留 content/category/prompt 与官方文件名规则。全部官方 evaluation 数据默认 `split=test`，不自动切一部分出来训练外循环。详见 `docs/BENCHMARKS.md`。

## 4. Planner / Verifier / Controller

三者角色分离，但不强迫部署三个框架。传输层使用配置式 Chat Completions JSON 接口。实际服务必须支持 `image_url`；音视频验收还必须支持同一请求中的真实 `input_audio` WAV。不是所有兼容接口、模型都支持该组合，必须先做能力检查。

配置 `ARAV_CONTROLLER_BASE_URL/MODEL/API_KEY`、`ARAV_VERIFIER_BASE_URL/MODEL/API_KEY`。只有已验证声音输入有效时才设置 `ARAV_VERIFIER_AUDIO=1`。环境变量中放密钥，不提交 `.env`。提示词在 `arav/prompts/`，也可以直接交给 Codex/已有 agent 使用。

```bash
python -m arav.cli plan --task configs/task.video.json --out runs/frozen-plan.json
```

人工检查并冻结计划。四个对比条件使用相同初始计划、原始任务、模型与种子：`vanilla` 不检查；`retry` 检查后只换 seed；`feedback` 根据反馈改当前 prompt；`evolved` 在反馈修复时使用已验证规则。当前版本规则用于失败后的修复，不宣称已经实现主动预测所有错误。

任务阶段不等于一个模型块：未完成但推进正确的块允许提交；只有 stage_complete 且达到最少块数才进入下一阶段。nominal/max_chunks 是显式可调原型参数，不是从论文推出来的最优超参数。

## 5. LongLive 纯视频参考后端

先按 NVlabs/LongLive 的 **LongLive1.0** 说明准备环境、权重和路径。复制修改 `configs/longlive.example.json`。只支持审查过的 Wan1.3B 16×60×104 latent 形状和 16 fps；其他分辨率/LoRA/2.0 不是即插即用。

```bash
python -m arav.cli run --task configs/task.video.json --plan configs/plan.video.json \
  --backend longlive --config configs/longlive.local.json --method feedback \
  --out runs/longlive-feedback-seed0 --seed 0
```

模型只在 worker 初始化时加载一次，候选重试不重新加载。通过 Python `run()` 多次调用并复用 driver，也能做持久化批量 worker；CLI 当前一次运行一个任务。prompt 改变时用已生成 latent 重建相关 KV，而非重生成旧视频。

**参考解码器当前每次无状态解码已选 latent 前缀，再只保存新帧**，因此保证旧 RGB 文件不被覆写，但耗时可能较高、拼接边界仍需实测。这是正确性基线，不能直接拿它宣称低开销流式生成。KV 快照也采用完整 clone，优化前先做性能剖析。GPU 性能实验必须计入这些开销。

## 6. 原生联合 AR 音视频接入

```bash
python -m arav.cli run --task configs/task.av.json --plan configs/plan.av.json \
  --backend my_native_av_driver:build --config configs/my_native_av.local.json \
  --method feedback --out runs/native-av-feedback-seed0
```

`my_native_av_driver` 是需要在真实模型代码仓实现的接口，不是本包内置模型。完整契约见 `docs/NATIVE_AV.md`：必须同时维护视频/音频 latent、两侧 KV 和 cross-attention、时间索引、解码器缓存与采样器状态；返回实际 MP4、按时序抽出的图片和 PCM WAV；失败时两路一起恢复。延迟生成的音频/视频必须等同一时间区间两路都齐全再提交。

公开 Talker-T2AV 是原生 AR 音频+说话运动的候选，但范围是说话头像，不能替代开放场景的拿杯/开门/叙事任务。MiniMax H3 的原生声音能力也不自动证明目标 checkpoint 暴露可复用的因果逐块 API。不要只把完整生成函数套一层重试就声称完成原生 AR 接入。

## 7. 最终导出与独立评分

```bash
python -m arav.cli assemble runs/longlive-feedback-seed0
```

导出不改 committed 文件。视频流连接时要求一致编码设置；音频需一致 PCM 格式，导出 MP4 时一次编码 AAC。失败但有部分前缀的任务也可导出，`complete_execution=false`，不能当成功样本过滤掉。

主表使用官方 evaluator。包内另提供 `judge-proxy`：独立模型只看原始任务、提前冻结的 reference events、最终媒体，不读取在线自评或方法身份。它只是一项可审计的 sampled-frame 项目指标，**不是官方 StoryEval / TempAct / AVGen / CineScope 分数**。音视频同样要真实声音输入。长视频的稀疏帧评审可能漏掉短动作，必须配合官方协议和人工抽查。

## 8. 外循环

`propose-rules` 只接收真实 TRAIN 轨迹；`validate-rules` 要求独立 DEV 的同任务/种子配对数据、同一旧策略版本、独立终局评分和真实总成本。按任务聚类 bootstrap，不把同任务多个 seed 当独立任务。准入判据预先固定；验证失败保留旧策略；最终 TEST 只读冻结策略。

```bash
python -m arav.cli propose-rules runs/train-a runs/train-b --out runs/proposal.json
python -m arav.cli validate-rules --proposal runs/proposal.json --pairs runs/dev-pairs.json --out runs/policy-v1.json
```

配对数据格式在 `docs/EXPERIMENTS.md`。这里只实现文本规则提案和离线准入，不宣称已经学到了更好的策略，也不会自动消耗 GPU 生成验证集。反复用同一 DEV 挑补丁会过拟合；必须保留最终未触碰测试集。

## 9. 下一位实现者入口

将 `docs/CODEX_HANDOFF.md` 交给 Codex。优先完成一个真实原生 AV 驱动的单块等价性、拒绝恢复和音视频时间对齐测试，再启动小样本真实实验。不要修改原有实验结果，不生成任何“看起来合理”的论文数据。
