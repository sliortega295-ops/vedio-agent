Review TRAIN-split execution traces and propose a bounded continuation policy patch.
Return one JSON object with a rules array (1..16 entries). Each entry contains
rule_id, rule (one concise reusable instruction), failure_type,
supporting_task_ids (array), mechanism_hypothesis, disconfirming_cases (array).
Use only supplied observations and measured outcomes. A repair followed by success
is a hypothesis, not causal proof; acknowledge seed luck and failed repairs.
A patch may clarify current-stage prompts or suggest resampling vs reprompting.
It cannot change original tasks, committed prefixes, verifier thresholds, metrics,
evaluation prompts, test splits, model weights, or budgets. Do not memorize a test
story or produce a general-purpose agent framework. No automatic deployment.
The harness will require paired frozen DEV runs, disjoint from the source cases,
and will deploy only after its prespecified validation gate passes. TEST evidence
must never be used to propose or validate a rule. Synthetic/mock runs are ineligible.
