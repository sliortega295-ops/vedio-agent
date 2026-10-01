# Source provenance / licenses

The previous repository payload is unchanged. No pretrained weights, external
benchmark dataset, or third-party evaluator implementation is vendored here.
Downloaders retain upstream README/license when available; users must review
model/data terms separately before use and redistribution.

LongLive adapter API and denoising/cache orchestration were reviewed against
NVIDIA's Apache-2.0 code:
https://github.com/NVlabs/LongLive/blob/main/LongLive1.0/pipeline/interactive_causal_inference.py
Source notice: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
The adapter is a new integration wrapper, not a claim of upstream endorsement.
Apache-2.0 license text is included at `licenses/Apache-2.0.txt` and available from
https://www.apache.org/licenses/LICENSE-2.0 .

The AVGen filename normalization follows its MIT-licensed generation/runner.py:
https://github.com/microsoft/AVGen-Bench . Copyright (c) Microsoft Corporation.
No complete evaluator is copied. Preserve its notices when using the benchmark.
MIT text is included at `licenses/MIT-AVGen.txt`.

Primary project sources:
- CineForge: https://arxiv.org/abs/2608.29621
- TempAct: https://github.com/jingw193/TempAct
- StoryEval: https://github.com/ypwang61/StoryEval
- AVGen-Bench: https://github.com/microsoft/AVGen-Bench
- LongLive: https://github.com/NVlabs/LongLive
- Talker-T2AV: https://github.com/zhenye234/Talker-T2AV

No license statement in this file changes the license of the original repo.
