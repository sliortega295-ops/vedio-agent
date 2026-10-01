# Benchmark 准备与使用边界

核验日期：2026-10-01。`arav/benchmarks.py` 保存可运行的下载/解析逻辑。本文所列可获取性来自官方仓库；完整数据未被伪装为已经随包附带。下载后以 manifest 的实际样本数为准。

| Benchmark | 真实入口 | 在本项目的角色 | 本次准备状态 |
|---|---|---|---|
| StoryEval | https://github.com/ypwang61/StoryEval · `prompts/all_prompts.json` | 连续事件执行；保留原始事件与视频名 | 数据下载器、离线解析器、输出映射已写；完整数据需运行下载 |
| TempAct | https://github.com/jingw193/TempAct · `dataset/temporal_eval_simple_100.csv` / `temporal_eval_combined_100.csv` | 时序指令遵循，纯视频 | 两份已核实 CSV 路径；官方工具在 `tools/` 和 `tools/metric/` |
| AVGen-Bench | https://github.com/microsoft/AVGen-Bench · `prompts/*.json` | 原生 AV 主候选：声音、同步、任务语义 | 自动枚举类别；输出 `category/safe_filename(content).mp4`；需安装官方多环境 evaluator |
| CineScope | https://arxiv.org/abs/2608.29621 | 长故事因果状态重点分析，另报完整叙事维度 | 尚未核实完整官方数据/评分包；程序明确报 blocked，不生成假官方数据 |
| VBench-Long | https://github.com/Vchitect/VBench | 辅助视觉质量；不替代任务完成率 | 仅预留手动附加评测，不声称封装完成 |

## 正式 evaluator

下载 manifest 记录了固定上游 commit。评分代码也应 checkout 相同 commit，保留模型版本、依赖、抽帧和聚合规则，不随意将自己的 VLM 问答打分称为官方分数。

StoryEval 使用上游 `evaluate.py` / `evaluate.sh`、`summarization.py`；先按 README 安装评审模型。TempAct 使用上游 `tools/eval_qwen3_multi.py` 或 `tools/eval_gemini.py` 及对应指标脚本。具体参数跟随已锁定版本 README，不假设任意后端返回格式一致。

AVGen 的环境是模块分开的 `environments/*.yml`，不能直接并成一个环境。整体流程入口是 `run_full_evaluation.sh` 和 `aggregate_score.py`；执行前审查脚本中的路径、模型名称、费用及所需权重。同步和唇同步不是同一指标：官方 README 的 `AV` / `Lip` 为越低越好，不得与其他越高越好的指标直接求均值。

## 协议

同一测试任务、同一 seed、同一初始计划、同一模型，对比四种内外循环设置。保存所有分配的 task ID 清单，失败/缺失输出保留在总体分母。自建长任务、把视频 prompt 增补声音、从 CineScope 筛连续镜头都必须声明为 derived 设置，不与原论文原始总分混为一谈。

经验学习使用另行构建并冻结的 TRAIN/DEV，且按完整故事和近重复模板分组；官方 benchmark 测试集不进入经验库。代码做 ID 隔离检查，但近重复文本/同源故事仍需实验组织者额外审核。

主结论至少同时报告：官方任务分数、全部事件按顺序完成比例、视觉质量、真实音视频同步与声音任务分数、失败率、总端到端成本。外循环成本另报，说明在多少新请求中摊销。内部 trace 的 verifier 接受率是过程量，不是正式任务成功率。
