# Decision 0003: Lock Experiment Semantics and Causal Metrics

- Date: 2026-08-02
- Status: Accepted

## Decision

Use [`EXPERIMENT_SPEC.md`](../EXPERIMENT_SPEC.md) as the normative tensor- and metric-level contract for the intervention study.

The released monitor is defined as the one-layer logistic probe applied to response-token `resid_post[12]`: the output of zero-based Gemma block 12, before block 13's input normalization. “Layer 12” selects the model block; compact probe key `0` indexes the one selected activation layer.

Use the ratio of paired dataset means as the primary rescue metric:

```text
Recovery(A) = [S_P(T_c ← N; A) - S_T] / [S_N - S_T]
```

Use the corresponding reverse-direction induction metric for sufficiency tests. Keep both metrics unclipped, report their component scores and denominator, preserve example pairing, and treat recovery as mechanistically interpretable only when the baseline suppression gap is positive.

Treat localization, necessity, sufficiency, cross-concept sharing, completeness, and behavioral preservation as separate claims with separate evidentiary requirements.

## Rationale

The released code uses three numbering systems that are easy to conflate: zero-based model blocks, the Hugging Face hidden-state tuple offset by one, and the compact selected-layer axis consumed by the probe. Locking the measured tensor prevents off-by-one and pre/post-block mistakes.

A normalized recovery fraction makes effects comparable across concepts with different raw probe-score ranges. Defining the reverse intervention and claim boundaries prevents a rescue result from being overstated as sufficiency, sharing, completeness, or preserved behavior.

## Consequences

- Day 3 implementation and tests must use the hook names, condition notation, response-relative token alignment, and metrics in the specification.
- Any replacement aggregation, tensor location, or claim criterion must be documented before component search and must supersede this decision explicitly.
- Component-level results must retain raw condition scores so normalized recovery can be audited.
