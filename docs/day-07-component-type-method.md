# Day 7 Attention-versus-MLP Method

This document implements the Day 4 freeze, retained-layer [Decision 0008](../decision-log/0008-retain-layers-09-through-12.md), and pre-result [Decision 0009](../decision-log/0009-freeze-day-07-component-type-procedure.md).

## Sites and execution

The selected layer order is `[12, 11, 10, 9]`. At each layer, the evaluator captures and patches all response tokens at `attn_out`, `mlp_out`, and `resid_post` exactly as named in `EXPERIMENT_SPEC.md`. The normal, correct-trigger, and irrelevant-trigger conditions reuse identical response IDs.

The monitor reads `resid_post[12]`. The evaluator therefore executes only through block 12 and raises a private early-termination signal after scoring that tensor. A real-checkpoint preflight compares normal, correct-trigger, and irrelevant-trigger scores with complete 42-block forwards and requires exact equality. It then requires zero score change for 36 same-condition identity patches.

## Evaluation grid

| Grid | Examples | Sites | Directions |
|---|---|---|---|
| Selected correct-trigger | All 1,408 benign examples: discovery and validation, positive and negative | 3 types × layers 12, 11, 10, 9 | Rescue and induction |
| Selected irrelevant-trigger | All 704 positive benign examples | Same 12 selected sites | Normal-to-irrelevant rescue |
| Random-layer control | 256 discovery-positive examples | 3 types × deterministic layers 5, 0, 8, 3 | Normal-to-correct rescue |

Every example also receives normal, correct-trigger, and irrelevant-trigger baselines. The expected raw result has 49,536 example/condition/intervention rows. Validation is used only for evaluation, never to change identities. Safety is never loaded.

## Metrics

For concept `c`, let positive means under normal and correct trigger be `N⁺` and `T⁺`, with stable denominator `D = N⁺ - T⁺ > 0`.

```text
positive rescue      = (P⁺_rescue - T⁺) / D
positive induction   = (N⁺ - P⁺_induction) / D
negative rescue shift    = (P⁻_rescue - T⁻) / D
negative induction shift = (N⁻ - P⁻_induction) / D
irrelevant rescue control = (P⁺_irrelevant - I⁺) / D
random-layer rescue control = (P⁺_random - T⁺) / D
```

Ratios remain unclipped. Negative shifts near zero support class-specific intervention effects; large absolute values flag broad monitor movement. Irrelevant and random values near zero support trigger and layer specificity.

## Uncertainty and comparisons

Use 10,000 paired percentile-bootstrap replicates with seed 42. Resample complete positive and negative example rows separately within each concept. Each positive resample supplies the denominator to both class strata. Equal-concept macro summaries are reported separately for discovery, held-out validation, and all benign concepts.

For every selected layer, role, class, and direction, report attention, MLP, and block effects plus paired attention-minus-MLP and block-minus-branch contrasts. `attention + MLP` is also reported descriptively, but it is not assumed to reconstruct the block effect because the branches occupy different sequential sites.

## Reproduce

```bash
external/minimal_neural_chameleons/.venv/bin/python \
  scripts/day07_run_component_types.py --batch-size 8

external/minimal_neural_chameleons/.venv/bin/python \
  scripts/day07_analyze_component_types.py

external/minimal_neural_chameleons/.venv/bin/python \
  scripts/day07_verify_component_types.py
```

The evaluator writes an ignored, resumable JSONL. The analyzer packs it into a deterministic gzip archive for commit and audit. The evaluator fails if a safety-unlock file exists and imports only discovery and validation through the guarded loader.
