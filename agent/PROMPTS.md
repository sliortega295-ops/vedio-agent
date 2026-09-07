# Prompt Inventory

SparseMAC keeps the current method prompts as package data and retains older
prompts in their original modules or files for provenance. This inventory is
the completeness checklist for the public agent package.

## Current Method: SparseMAC-SemanticV2

| Stage | Source | Purpose |
|---|---|---|
| Planner | `sparsemac/prompts/semantic_v2/planner.txt` | Creates four generation-ready prompt variants and 5-10 frozen, weighted, atomic semantic criteria from the immutable original prompt. |
| Verifier | `sparsemac/prompts/semantic_v2/verifier.txt` | Scores a sparse or full storyboard against those frozen criteria without visual-quality penalties. |
| Analyzer | `sparsemac/prompts/semantic_v2/analyzer.txt` | Uses structured QA evidence and round history to decide whether to refine and to produce complete rewritten prompts. |
| Client system message | `sparsemac/luna_client.py`, `LunaClient.generate_json` | Requires one precise JSON object from the OpenAI-compatible model. |

These three stage templates are loaded by `sparsemac/semantic_v2.py`; they are
the prompts used when `--agent-method sparsemac-semantic-v2` is selected.

## Legacy PlanV4 and Verification

| Stage | Source |
|---|---|
| Legacy planner | `sparsemac/planner.py`, `PLANNER_PROMPT_TEMPLATE` |
| Archived DeepSeek planner text | `sparsemac/prompts/planner_deepseek_v1.txt` |
| Legacy analyzer | `sparsemac/analyzer.py`, `ANALYZER_PROMPT` |
| Legacy API verifier | `sparsemac/verifier.py`, `VERIFY_PROMPT` |
| Legacy LLaVA verifier | `sparsemac/llava_client.py`, `LLAVA_VERIFY_QUERY` |
| Legacy visual judge | `sparsemac/visual_judge.py`, `VISUAL_JUDGE_PROMPT` |
| Orchestrator storyboard judge | `scripts/sparse_to_full_vlm_pipeline.py`, `STORYBOARD_API_PROMPT_TEMPLATE` |
| Orchestrator refinement | `scripts/sparse_to_full_vlm_pipeline.py`, `DEEPSEEK_REFINE_PROMPT_TEMPLATE` |
| Resume storyboard judge | `scripts/resume_sparse_vlm_pipeline.py`, inside `robust_vlm_eval` |

## Historical Mixed-Skill Planner

The mixed-skill experiment is not used by SemanticV2, but all its prompt cards
are included:

- router: `sparsemac/prompts/mixed_skill_router.txt`;
- shared planner: `sparsemac/prompts/mixed_planner_base.txt`;
- primary cards: `sparsemac/prompts/primary_skills/*.txt`;
- secondary cards: `sparsemac/prompts/secondary_skills/*.txt`.

Both card directories contain the seven T2V-CompBench semantic categories:
`consistent_attr`, `dynamic_attr`, `spatial_relationship`, `motion_binding`,
`action_binding`, `interaction`, and `numeracy`.

## Dense-Only Ablation

`scripts/planv4_dense4_full_refine.py` includes the ablation's `VLM_PROMPT` and
`REFINE_PROMPT`. They are intentionally separate from the sparse agent prompts
so the dense-only control remains auditable.

## Prompt Integrity Rules

- The original benchmark prompt is immutable.
- SemanticV2 criteria are generated once and frozen for all rounds.
- Prompt variants may clarify execution but must not add benchmark facts.
- The verifier sees the original prompt, frozen criteria, candidate prompt, and
  storyboard; the analyzer sees structured evidence rather than raw images.
- No API key, endpoint credential, generated response, or benchmark answer is
  embedded in a prompt asset.
