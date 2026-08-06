# Day 9 Grouped Necessity and Completeness Method

This procedure implements the Day 04 freeze, the Day 08 component order, and [Decision 0012](../decision-log/0012-freeze-day-09-grouped-necessity-procedure.md). The machine-readable group definitions are frozen in `results/day-09/frozen-group-plan.json` before grouped model results.

## Intervention sets

The primary curve uses nested prefixes 1, 2, 4, 8, and 16 of the committed Day 08 selected order. The matched control curve uses the same sizes from the committed deterministic random-control order. Both are response-token normal-to-correct-trigger grouped patches.

Two diagnostics supplement those curves:

- `all_layer11_components`: all 16 pre-output-projection query heads plus the whole MLP at zero-based layer 11;
- `control_outside_layer11_k17`: 17 deterministic nonselected candidates outside layer 11, selected by the frozen SHA-256 rule.

The size-17 control is not component-type matched because every eligible whole-MLP candidate is already in the selected top 16. `resid_post_layer12_positive_control` replaces the complete measured block output and should recover exactly one on positives.

## Probe-score grid

Evaluate all 1,408 benign examples and both labels. For each example record normal and correctly triggered baselines plus 13 rescue interventions: five selected prefixes, five random prefixes, the layer-11 complete candidate group, its size-17 control, and the block-12 positive control.

For concept c and group G:

    positive recovery = (mean patched positive - mean triggered positive)
                        / (mean normal positive - mean triggered positive)

    negative shift    = (mean patched negative - mean triggered negative)
                        / positive suppression denominator

The primary curves report equal-concept macros separately for discovery and validation. Confidence intervals use paired example resampling within concept/class, 10,000 replicates, and seed 42.

## Behavior grid

Reuse the Day 08 subset selected by ascending SHA-256 of `day08-behavior:{example_id}` within each concept/class cell. For all 13 interventions, run the complete model and calculate patched minus correctly triggered destination response-token NLL. This is a bounded damage diagnostic, not a complete behavioral-preservation test.

## Compactness classification

For each selected prefix calculate recovery divided by the directly measured selected K=16 recovery within discovery and validation. Use point estimates and the frozen rule:

- compact: K<=4 reaches 80% in both scopes;
- moderately distributed: K=8, but not K<=4, reaches 80% in both scopes;
- highly distributed: K=8 fails 80% in either scope.

Report K=16 recovery, all-layer-11 recovery, random controls, negative shifts, and behavior NLL separately. The relative label does not claim that K=16 is absolutely complete.

## Numerical controls and execution

CPU tests require exact group-versus-independent agreement, group-order invariance, and no leaked hooks. Real-checkpoint preflight requires exact same-shape identities. Changed-batch vector execution is limited to two grouped jobs and must agree with independent probe scores within 0.002 and response NLL within 0.02.

The evaluator fails closed if a safety-unlock file exists. Uncompressed JSONL files are ignored resumable state; deterministic gzip archives, summaries, tables, figures, and audits are tracked.

## Execution and verification

The frozen plan was committed at `7fa5c9c` before any grouped result. The implementation was committed at `ff2328a` before full execution. Run from the repository root with the recorded project environment:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src \
  external/minimal_neural_chameleons/.venv/bin/python \
  scripts/day09_run_grouped_necessity.py

PYTHONPATH=src external/minimal_neural_chameleons/.venv/bin/python \
  scripts/day09_analyze_grouped_necessity.py

PYTHONPATH=src external/minimal_neural_chameleons/.venv/bin/python \
  scripts/day09_verify_grouped_necessity.py
```

The completed evaluator produced exactly 19,712 probe-score rows and 616 behavior rows. The real-checkpoint preflight passed 26 exact identity checks, exact vectorized score and NLL comparisons, exact group-member-order invariance, and hook cleanup. The independent audit passes 21/21 checks, all 37 repository tests pass, and a second analysis run reproduced the raw archives, summaries, PDFs, and manifest byte for byte.

The complete outcome and interpretation are in [results/day-09](../results/day-09/README.md).
