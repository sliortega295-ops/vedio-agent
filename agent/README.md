# SparseMAC Agent Pipeline

This directory contains the complete SparseMAC orchestration layer used for
prompt planning, sparse candidate search, semantic VLM verification, iterative
prompt refinement, warmup-latent reuse, and final full-video selection. It is
separate from the frozen candidate-concurrency experiment under `package/`.

The current method is `sparsemac-semantic-v2`. Legacy planners, evaluators, and
dense-only ablations are retained so older runs remain interpretable.

## Pipeline

```text
original prompt
  -> planner: frozen weighted semantic criteria + four prompt variants
  -> four independent sparse Wan candidates (prompt + seed)
  -> storyboards + semantic VLM scoring
  -> analyzer: structured failure analysis and complete prompt rewrites
  -> repeat sparse search until threshold or round limit
  -> top two sparse candidates resume from their own warmup latents
  -> two full 81-frame videos + final semantic VLM scoring
  -> selected final video and pipeline report
```

The planner runs once. Its 5-10 atomic semantic criteria are frozen across all
candidates, refinement rounds, and final-video review. The verifier scores only
prompt-semantic alignment; it does not add visual-quality questions or penalize
normal sparse-preview blur and missing in-between frames. The analyzer receives
structured QA evidence, not raw images, and rewrites a complete executable
generation prompt when refinement is useful.

## Included Code

- `scripts/sparse_to_full_vlm_pipeline.py`: primary single-prompt orchestrator.
- `scripts/run_t2vcompbench50_agent.py`: balanced T2V-CompBench batch runner.
- `scripts/resume_sparse_vlm_pipeline.py`: resumes incomplete historical runs.
- `sparsemac/planner.py`: current and legacy planning logic.
- `sparsemac/semantic_v2.py`: current criteria, verifier, and analyzer contract.
- `sparsemac/generator.py`: sparse/full process launcher and GPU assignment.
- `sparsemac/luna_client.py`: OpenAI-compatible text/vision JSON client.
- `sparsemac/vision.py`: frame extraction, storyboards, and legacy CV metrics.
- `generation_backend/`: Wan sparse generation and warmup-latent full resume.
- `sparsemac/prompts/`: every external prompt template and legacy skill card.
- `PROMPTS.md`: complete inventory of external and embedded prompts.
- `scripts/generate_planv4_dense4.py` and
  `scripts/planv4_dense4_full_refine.py`: dense-only PlanV4 ablation.
- `scripts/run_mixed_skill_plan_eval.py`: historical mixed-skill planner study.

## Install

Create a private environment. Install a CUDA-compatible PyTorch build first,
then install the agent and generation extras:

```bash
conda create --prefix "$HOME/conda_envs/sparsemac" python=3.10 -y
conda activate "$HOME/conda_envs/sparsemac"

# Install the PyTorch build appropriate for the host CUDA driver first.
python -m pip install -e './agent[generation,dev]'
```

The bundled generation backend imports its private minimal Wan source from
`agent/generation_backend/wan`. Set `SPARSEMAC_WAN_SOURCE_ROOT` only when using a
different compatible Wan checkout. Model weights are not included.

## Configure

Use `config/env.example` as the variable reference. Export credentials in the
process environment or place a populated file at `agent/.env.local`; that file
is ignored by Git.

Required for the current method:

```bash
export SPARSEMAC_LUNA_API_KEY='...'
export SPARSEMAC_LUNA_BASE_URL='https://your-endpoint.example/v1'
export SPARSEMAC_LUNA_MODEL='gpt-5.6-luna'
export SPARSEMAC_WAN_CHECKPOINT='/path/to/Wan2.1-T2V-1.3B'
export SPARSEMAC_PYTHON="$CONDA_PREFIX/bin/python"
```

The VLM endpoint must implement OpenAI-compatible chat completions with image
data URLs and JSON-object responses. Do not commit populated API credentials.

## Run One Prompt

From the repository root:

```bash
python agent/scripts/sparse_to_full_vlm_pipeline.py \
  --remote local \
  --agent-method sparsemac-semantic-v2 \
  --prompt 'A red car drives from left to right, then stops by a blue house.' \
  --candidates 4 \
  --max-rounds 3 \
  --sparse-warmup-ratio 0.10 \
  --sparse-gpus 0,1,2,3 \
  --full-gpus 0,1
```

Repository-local defaults are used for the project, generation backend, logs,
and runs. Every machine-specific path can be replaced with a CLI flag or the
environment variables in `config/env.example`.

For orchestration on a separate configured host, pass `--remote HOST` together
with remote values for `--project-dir`, `--generator-root`, `--python-bin`,
`--ckpt-dir`, and `--run-root`. The API variables must be available in the
remote process environment.

## Run T2V-CompBench

Supply a T2V-CompBench prompt directory containing the seven official category
text files, then run:

```bash
python agent/scripts/run_t2vcompbench50_agent.py \
  --remote local \
  --prompt-dir /path/to/T2V-CompBench/prompts \
  --agent-method sparsemac-semantic-v2 \
  --prompt-count 50 \
  --candidates 4 \
  --max-rounds 3
```

This script generates a fixed benchmark manifest before execution and writes a
JSON/CSV batch summary. Official T2V-CompBench scoring is a separate evaluation
stage; VLM selection scores must not be reported as official benchmark scores.

## Outputs

A run under `agent/runs/<run_id>/` contains:

- `plan.json`: immutable original prompt, variants, and frozen criteria;
- `round_<n>/candidates.json`: prompt, seed, video, and warmup-latent metadata;
- `round_<n>/sparse_visual_judge.json`: structured sparse VLM evidence;
- `round_<n>/vlm_refine_plan.json`: analyzer decision and next prompts;
- `round_<n>/final_cand_*/`: full videos and final judgments;
- `timings.json`: planner, generation, VLM, and wall-clock timings;
- `pipeline_report.json`: selected sparse/full candidate and method metadata.

Generated videos, model weights, latent checkpoints, logs, and credentials are
excluded from Git.

## Reproducibility Boundary

The committed implementation is code-complete for the Wan SparseMAC path, but
a real GPU run still requires compatible Wan checkpoints, CUDA/PyTorch, ffmpeg,
and a working VLM endpoint. The optional legacy LLaVA and Gemini paths require
their own environments. The exact prompt sources are documented in
`PROMPTS.md`; no prompt is fetched implicitly at runtime.
