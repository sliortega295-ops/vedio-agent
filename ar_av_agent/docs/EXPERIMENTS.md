# v0 实验预案（未执行模型实验）

首先同一真实后端、同一初始计划与种子对比 vanilla、retry、feedback、evolved。另行实现整段 Best-of-N 才能声称比完整重试更省；当前版本没有冒充该基线已经实现。比较需报告相同预算曲线以及达相同质量的总成本，而不是仅比较单次成功采样。

Pilot 建议先人工冻结少量易于观察的多阶段任务，检查原模型是否存在足够错误空间、verifier 是否漏检/误拒、当前 chunk 重试是否确实有效。之后再扩大官方数据集规模。不要预先填写任务分数或加速比例。

每次生成记录模型加载、KV 快照、采样+解码、验证、反馈和恢复成本；最终导出与独立评分另有开销。外部模型调用保留 usage/token 元数据与耗时。没有测 GPU 活跃时间时不要把 wall time 改名成 GPU seconds。失败采样和初始化都不能隐去；批量运行可复用同一个 driver，但必须分别报告摊销和非摊销口径。

官方 benchmark 衡量最终输出；内部 `completed_stage_ids` 不是最终任务成功标签。proxy 的未知判为未证明成功并报告判断覆盖率，正式报告另人工核验。相同视觉帧验证器不够证明高精度唇同步，须调用官方音视频同步模型。

## 规则提案 / 配对 DEV 准入

`propose-rules` 仅从真实 train trace 提案。`validate-rules` 不替你生成配对视频；它对已执行、独立评分的 DEV 记录做准入。每一项记录具有以下结构（数值必须替换为实测，不提供示例实验成绩）：

```json
{
  "task_id": "heldout-dev-task-id",
  "seed": 0,
  "split": "dev",
  "mock": false,
  "old_policy_hash": "SHA256_OF_OLD_RULE_LIST",
  "new_policy_hash": "SHA256_FROM_PROPOSAL",
  "independent_judge": "FROZEN_EVALUATOR_VERSION",
  "old_score": null,
  "new_score": null,
  "old_total_s": null,
  "new_total_s": null
}
```

null 会被程序拒绝，必须补齐同一个评分协议归一到 [0,1] 的实测结果。新旧 score 的比较必须固定评价模型、原始任务与媒体协议。生成失败也保留在配对中，并按预先规定的失败定义判分、计费。

默认至少 20 对、至少 10 个不同 DEV 任务，按 task 聚类 bootstrap；质量差区间下界不小于 -0.02，同时成本差区间上界小于 0 才部署。阈值是原型的预注册候选，不是已有论文结论。多个补丁反复筛选须控制选择偏差，最终 TEST 不触碰。

策略文本摘要的 hash 必须与验证记录一致；TEST ID 不能出现在训练或 DEV 里。代码检查不了人工伪造记录或语义近重复，需要审计原始视频、日志、task lineage。mock测试里的数值仅用于单元测试算术，绝不属于实验证据。
