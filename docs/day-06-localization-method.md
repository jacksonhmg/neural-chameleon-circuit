# Day 6 Coarse Residual-Stream Localization Method

This method implements the frozen Day 4 coarse-layer rule and [Decision 0007](../decision-log/0007-freeze-day-06-localization-procedure.md). It is fixed before inspecting any Day 6 intervention result.

## Scope

- Model: released Neural Chameleon at revision `c4aeb57502a8bed6a62dac57819eec27fd5e3c99`.
- Data: all 256 positive examples from the four discovery concepts, with 64 examples per concept.
- Forbidden data: validation and safety examples and all historical result files.
- Site: raw `resid_post[L]` for zero-based blocks 0–41.
- Directions: normal-to-triggered rescue and triggered-to-normal induction.
- Teacher forcing: exact shared response-token IDs in every condition.

## Token regions

The rendered prompts have different lengths. For each example, remove left padding and compare normal and triggered prompt-token IDs with `difflib.SequenceMatcher(autojunk=False)`. Concatenate every nonempty matching block in order. The resulting exact-ID index pairs define `prompt`; unmatched tokens—including the trigger and any changed boundary token—are not patched.

`response` contains every valid response-relative offset. `all_aligned` is the disjoint union of `prompt` and `response`. Padding is never patched. Raw rows record the number of patched tokens and prompt-alignment coverage.

## Execution boundary and identity controls

Only blocks 0–12 can causally affect the probe's `resid_post[12]` input. Executable interventions stop immediately after the block-12 hook has observed the patched state; a regression test and a real-checkpoint smoke test require exact agreement with the score from a full 42-block forward pass.

Blocks 13–41 are included in all summaries and heatmaps with patched score equal to the relevant destination baseline, recovery zero, and induction zero. They are labeled `structural_causal_null`, not represented as measured forward passes.

Before the full scan, the first deterministic length-sorted discovery batch is identity-patched for normal-to-normal and triggered-to-triggered at every executable layer and token region. Every score must be bit-identical to its unpatched baseline.

## Metrics and uncertainty

For concept `c`, let `N_c` and `T_c` be the normal and triggered positive-example means and `P_c` a patched mean. The suppression denominator is `D_c = N_c - T_c` and must be positive.

```text
Recovery_c = (P_rescue,c - T_c) / D_c
Induction_c = (N_c - P_induction,c) / D_c
```

Ratios are not clipped. Macro effects give each discovery concept equal weight. Intervals use 10,000 paired percentile-bootstrap replicates with seed 42: independently resample complete example rows within each concept, retain every baseline and intervention together, compute each concept ratio, then take their equal-weight mean.

The inferential onset for a direction and region is the earliest executable layer whose macro 95% interval is strictly above zero. The frozen coarse ranking sorts all 42 layers by macro full-response rescue, breaking exact ties toward the lower layer, and retains four.

## Reproduce

```bash
external/minimal_neural_chameleons/.venv/bin/python \
  scripts/day06_run_localization.py --batch-size 4

external/minimal_neural_chameleons/.venv/bin/python \
  scripts/day06_analyze_localization.py

external/minimal_neural_chameleons/.venv/bin/python \
  scripts/day06_verify_localization.py
```

The evaluator is resumable by example/intervention key. It fails closed if the safety unlock file exists and imports only the discovery loader.
