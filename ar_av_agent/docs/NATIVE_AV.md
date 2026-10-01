# 原生音视频 driver 的严格契约

`arav.core.Driver` 是完整接口；`--backend local_module:factory` 导入用户明确选择的本地模型集成。factory 接收 JSON config，返回持久化对象。核心 runtime 不负责搬运模型权重，也不重新实现上游 sampler。

`reset()` 清空本条任务状态，不卸载模型。`snapshot()` 返回无可变别名的检查点。`sample(stage, control, seed, out, start_s)` 只推进 tentative 状态，输出 Candidate。`commit(candidate)` 推进提交边界；`restore(snapshot)` 恢复候选之前的状态，不修改任何已提交媒体。`synchronize()` 让实测耗时包含异步 GPU 执行。

必须显式审查的状态包括：视频/音频历史 latent、两路 self-KV、跨模态/文本 cross-KV、位置/窗口索引、文本条件版本、视频 VAE 特征缓存、音频 codec/vocoder 缓存、音频与视频不同采样时钟、分块重叠窗口、未对齐的延迟队列和 sampler RNG。模型冻结参数不要复制进 snapshot。

Candidate 包含当前 chunk 实际视频路径、按顺序的图像证据、PCM WAV、所有应跟踪文件、视频帧数/fps、音频样本数/sample_rate、两路起始时间。所有路径必须位于当前 run 目录，不能指向上一次候选的共享可覆写输出。每次重试写入独立目录。禁止把 prompt/ground-truth 标签塞进 observation 让 verifier 冒充看过媒体；该字段仅供 mock 测试。

如果模型先产音频，延迟若干步才产对应运动，两侧尚不完整的时间区间属于 tentative buffer，不准提前播放/提交一侧。改变 prompt 可能需要重新构建条件缓存；必须保持已选 latent 与已提交波形/视频不变。只处理新续写，不做历史回退。

## 接入验收

先固定 noise/seed，比较不接 controller 和 always-accept 控制下逐块输出、KV 位置及帧数是否一致。再注入一次拒绝，比较恢复后的下一次样本和独立从同一快照采样是否一致，覆盖视频+音频+解码器三部分。单独测试 prompt 切换、音频静音/缺轨、超长状态、OOM/超时、音视频不同时间长度和边界噪声。

这套测试通过后再声明一个后端 integrated。仅调用完整 `generate()` 然后拼多个结果不是该接入。当前 LongLive adapter 只能承担 video-only 调试。

## 已审查的候选与限制

- NVlabs/LongLive：https://github.com/NVlabs/LongLive ，参考接口是 `LongLive1.0/pipeline/interactive_causal_inference.py`，含条件切换 recache。本包只适配这一版固定形状，不把 LongLive-Plug/2.0 混用。
- Talker-T2AV：https://github.com/zhenye234/Talker-T2AV ，`speech_llm.py::generate_speech_motion` 有共享 AR 状态、speech/motion 两路 CFM 和 motion delay；定位为说话头像，非通用动作故事。未完成本包逐块接入和 LIA-X 渲染适配。
- MiniMax H3：https://huggingface.co/MiniMaxAI/MiniMax-H3 ，官方说明支持原生声音；**仍需审查具体 checkpoint 的因果分块采样器及联合缓存**，不能由“原生音视频”自动推断“可用于本实验的逐块 AR API”。

这里不指定一个未经验证的通用 AV 后端为“已经能跑”。后续选定真实模型后，只替换 driver，保留任务、提示词、控制与评价协议。
