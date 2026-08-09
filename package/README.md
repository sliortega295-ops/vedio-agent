# Test-Time Scaling: Random-8 / RoPE Ablation -> VLM Select -> T2V-CompBench

This package contains only the independent test-time-scaling experiment:

```text
T2V-CompBench prompts
        |
        v
Wan2.1-T2V-1.3B/14B: 8 independent seeds, dense or sparse-RoPE videos
        |
        v
VLM chooses one candidate per prompt
        |
        v
Official T2V-CompBench evaluation of the selected videos
```

It intentionally does **not** contain SparseMAC agent code, planner/analyzer
code, rewrite logic, or agent prompt templates. Model weights, Conda
environments, generated videos, and latent checkpoints are not included.
The requested VLM API configuration is stored in `config/vlm.env`.

## Contents

- `scripts/generate_wan_random8.py`: prepare a manifest and generate eight
  independent dense candidates per prompt. The Wan pipeline stays resident for
  one worker shard.
- `scripts/launch_random8.sh`: one generation process per visible GPU.
- `scripts/generate_sparse_rope8.py`: generate matched N6/N9 sparse candidates
  with gapped, compact, or affine temporal RoPE positions. One full-frame
  warm-up is shared across the three branches for each prompt/seed.
- `scripts/launch_sparse_rope8.sh`: one sparse RoPE worker per visible GPU.
- `scripts/select_vlm.py`: make storyboards, call the VLM, rank the eight
  candidates, and optionally materialize a one-candidate run directory.
- `scripts/evaluate_t2vcompbench34b.py`: official seven-category evaluator
  using the LLaVA-34B and CV evaluator split used in the experiment.
- `third_party/wan/`: Wan2.1 Python source needed by the generator.
- `third_party/T2V-CompBench/`: evaluator source and benchmark prompt/meta-data
  files. This is benchmark code, not agent code.
- `requirements/`: dependency specifications and environment notes.
- `config/vlm.env`: requested VLM endpoint/model/key configuration.

## 1. Prepare eight seeds

Use the packaged benchmark source or an equivalent T2V-CompBench checkout:

```bash
PKG=/path/to/test_time_scaling_8seed_vlm_t2vcompbench
RUN=$PWD/runs/random8-dense
PY=/path/to/conda/envs/wan_gen/bin/python

"$PY" "$PKG/scripts/generate_wan_random8.py" prepare \
  --compbench-root "$PKG/third_party/T2V-CompBench" \
  --output-root "$RUN" \
  --prompts-per-category 50 \
  --prompt-sample-seed 20260608 \
  --seed-master 20260609
```

This creates 350 prompts x 8 candidates = 2,800 video tasks. The generation
defaults match the recorded Wan1.3B dense baseline: `832x480`, 81 frames,
50 steps, shift `8.0`, CFG `6.0`.

The same generator supports the Wan 14B text-to-video config:

```bash
  --wan-model t2v-14B \
  --ckpt-dir /gfs/space/jiayu/models/Wan2.1-T2V-14B
```

HunyuanVideo-1.5-8B is listed in the model requirements for the comparison
set, but its upstream inference backend is intentionally not mixed into this
minimal Wan runner.

Launch one process per GPU. The script refuses no GPU isolation on its own, so
set the list explicitly and keep one GPU per process:

```bash
export RUN_ROOT="$RUN"
export CKPT_DIR=/data/models/Wan2.1-T2V-1.3B
export PYTHON_BIN="$PY"
export GPU_LIST=0,1,2,3,4,5,6,7
bash "$PKG/scripts/launch_random8.sh"
```

For a smoke test, use one category and a small prompt count during `prepare`,
then run with `GPU_LIST=0`.

## 2. Sparse RoPE test-time scaling

Prepare sparse manifests from the dense manifest. The same prompt/seed pool is
reused, so the RoPE comparison is paired:

```bash
SPARSE=$PWD/runs/rope-scaling
"$PY" "$PKG/scripts/generate_sparse_rope8.py" prepare \
  --dense-manifest "$RUN/manifest.json" \
  --output-root "$SPARSE" \
  --patterns n6_three_block n9_three_block \
  --variants gapped compact affine \
  --seeds-per-prompt 8
```

The default patterns are N6 `[0,1,2,9,10,11]` and N9
`[3,4,5,8,9,10,13,14,15]`. Each branch receives native VAE output: N6 has
21 decoded frames and N9 has 33 decoded frames. No interpolation or frame
repetition is used. Run each pattern with one worker per GPU:

```bash
export RUN_ROOT="$SPARSE"
export CKPT_DIR=/data/models/Wan2.1-T2V-1.3B
export GPU_LIST=0,1,2,3,4,5,6,7
export PATTERN=n6_three_block
bash "$PKG/scripts/launch_sparse_rope8.sh"
```

Run `select_vlm.py` separately on each pattern/variant manifest, then pass
each materialized one-candidate root to the official evaluator. The VLM sees
only the original prompt, optional frozen criteria, and storyboard; it does
not see the RoPE label or seed.

## 3. VLM selection

The selector sends only `original_prompt`, optional frozen semantic criteria,
and the storyboard image. It never sends the generation prompt or seed.

```bash
set -a
source "$PKG/config/vlm.env"
set +a

SELECT=$PWD/runs/random8-vlm-selection
BEST=$PWD/runs/random8-vlm-selected

"$PY" "$PKG/scripts/select_vlm.py" \
  --run-root "$RUN" \
  --output-root "$SELECT" \
  --materialize-root "$BEST" \
  --max-concurrency 4
```

`--max-concurrency` is capped at 4. By default, the selector uses the
original-prompt-only rubric embedded in `select_vlm.py`. To reproduce the
previous frozen-VQA scoring protocol, pass an external directory containing
precomputed JSON criteria:

```bash
  --criteria-root /path/to/frozen_criteria
```

The criteria directory must contain either
`<category>/<prompt_id>.json` or `<prompt_id>.json`, with a top-level
`criteria` list. The planner/verifier that produced those criteria is not part
of this package.

## 4. Official T2V-CompBench score

Run the evaluator on `$BEST`, which contains exactly one VLM-selected video per
prompt:

```bash
GEN_PY=/path/to/conda/envs/gen/bin/python
CV_PY=/usr/bin/python
LLAVA=/data/models/eval_weights/t2v_compbench/LLaVA/llava-v1.6-34b
CV_SITE=/path/to/t2vcompbench_cv_site
DEPTH=/data/models/eval_weights/t2v_compbench/DepthAnything/depth_anything_vitl14.pth
HF_HOME=/data/models/huggingface

"$GEN_PY" "$PKG/scripts/evaluate_t2vcompbench34b.py" \
  --run-root "$BEST" \
  --compbench-root "$PKG/third_party/T2V-CompBench" \
  --python-bin "$GEN_PY" \
  --cv-python-bin "$CV_PY" \
  --llava-model "$LLAVA" \
  --eval-site "$CV_SITE" \
  --depth-weights "$DEPTH" \
  --hf-home "$HF_HOME" \
  --llava-gpus 0,1 \
  --cv-gpus 2,3 \
  --stage all
```

The final files are written below `$BEST`:

- `summary34b.json`: seven-category summary;
- `scores34b/all_candidates.json`: normalized per-video scores;
- `selection34b/best_by_prompt.json`: official evaluator selection record;
- `benchmark34b/`: evaluator work directories and logs.

## Reproducibility boundary

The VLM score is a selection signal, not an official benchmark score. The
official score is computed only after materializing the selected dense video.
Do not compare a VLM score directly with a T2V-CompBench score without keeping
the two fields separate.
