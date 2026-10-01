# 可直接给 Codex 的实现提示词

你是本项目的实现与实验代理。先阅读 `ar_av_agent/README.md`、`arav/core.py`、`docs/NATIVE_AV.md` 和测试；仓库根部原有 package/results/docs 是另一个已冻结的稀疏候选并发实验，不要修改、重算或混入本项目结果。先检查当前 git diff，保留用户未提交变更。

目标：把 verified-prefix 原生自回归音视频 agent 做到真实模型可测。系统维护不可变已提交前缀，只生成下一段 tentative AV，验证通过才两侧一起 commit，失败只恢复本次采样开始状态、调整当前条件重试。禁止添加历史回退深度搜索、修改已提交内容、自动放宽验证标准或将后期配音称为原生联合 AV。

请按下列顺序执行，不需要重建通用 agent 平台：

1. 对用户可用的模型 checkpoint 检查实际 sampler，明确视频/音频是否都是因果分块，是否暴露 KV/latent continuation，是否具有不同时间率/延迟。不要由论文名称或多模态输入能力推断这些事实。若只能拿到视频模型，明确只跑 video-only pilot；若拿到头像模型，不用它做开放场景官方榜单。
2. 实现本地 `module:factory` driver。模型常驻、snapshot不复制权重、候选文件独立目录；保存两路 KV、cross-attention、文本条件版本、时间位置、音视频解码缓存和 RNG。先用稳妥的深拷贝做参考，再剖析是否值得优化。不先写复杂多卡调度器。
3. 先执行 CPU contract tests，再用真实模型做 always-accept 等价性、强制拒绝恢复、换 prompt recache、音视频时钟、候选与提交文件哈希测试。将环境、checkpoint hash、源码 commit、完整命令、错误记录到新的 run artifact；没有真实测量就写 Not Run。
4. 配置实际支持图像+声音输入的 Verifier，证明其收到真实波形，缺声音必须 unknown/reject。短 chunk 可以动作未完成但推进正确；阶段完成需检查后置条件。不要让每一个块都满足整个最终剧本。先人工审查一个小 pilot 的误接受与误拒绝。
5. 执行 benchmark 下载并锁定 manifest。所有 public test prompts 禁止写回规则库，StoryEval原始event_list不能被planner plan替换。CineScope未验证发布则保持blocked。需要派生长音视频任务时另起source名称，保存原始来源与改写差异。
6. 先运行 vanilla/retry/feedback 的配对 pilot；固定原始任务、初始计划、种子、生成器、预算和最终评价协议，记录所有失败。确认有真实收益空间后，才从独立TRAIN轨迹提出规则，在DEV上按冻结门槛验证，最后在未触碰TEST上测evolved。
7. 以最少必要模块交付补丁与结果。所有性能包括失败候选、decode、snapshot/restore、recache和VLM；分别报告冷启动、常驻模型和跨任务经验摊销成本。没有实测不能写加速倍率、任务分数或“解决漂移”。

最重要的交付是：一个真实原生 AV driver、通过的真实状态恢复测试、官方数据manifest、独立最终评分与完整预算记录。不把单元测试mock通过或控制器自评当论文结果，不假装已复现CineForge/SPIRAL/TBS/TempAct。
