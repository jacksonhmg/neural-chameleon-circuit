# Decision 0012: Freeze the Day 9 Grouped-Necessity Procedure

- Date: 2026-08-05
- Status: Accepted before grouped intervention results

## Decision

Carry forward the exact Day 8 discovery-frozen order without reranking. Test nested selected prefixes K=1, 2, 4, 8, and 16 and the corresponding nested prefixes of the 16 Day 8 deterministic random controls. Patch normal response-token activations into correctly triggered runs at every site in a group and measure the released block-12 probe.

Define the complete relevant-block diagnostic as all 16 pre-output-projection heads plus the whole MLP at zero-based layer 11. Compare it with a deterministic size-17 control chosen by ascending SHA-256 of `42:{candidate_id}` among candidates outside both layer 11 and the frozen top 16. Because all four eligible MLP candidates entered the frozen top 16, no nonselected MLP exists; the size-17 control therefore contains heads only, and conclusions must disclose that type-matching limitation.

Include complete `resid_post` replacement at layer 12 as the direct positive control. It defines complete recovery at the measured activation but does not localize the origin of the mechanism.

Run the complete grouped grid on all 1,408 benign discovery and validation examples, both positive and negative. Positive recovery uses the frozen suppression denominator. Negative-example score shift uses the same numerator orientation and positive denominator, so it is an off-target monitor movement rather than recovery. Run full-model response NLL for every grouped intervention on the same frozen 44-example, two-per-concept/class behavior subset used on Day 8.

Use two grouped jobs per expanded vector batch at most. Deterministic CPU tests must be exact. Real BF16/MPS preflight requires exact same-shape identities and no more than 0.002 absolute probe-score or 0.02 response-NLL disagreement between grouped vector and independent execution.

Use 10,000 paired percentile-bootstrap replicates with seed 42, resampling complete example rows within concept and class and weighting concepts equally. Do not assume monotonicity or add individual effects.

Classify the selected curve as compact when K<=4 reaches at least 80% of K=16 rescue on both discovery and validation macro point estimates; moderately distributed when K=8, but not K<=4, reaches that threshold on both; and highly distributed when K=8 fails the threshold in either scope. Report absolute K=16 and all-layer-11 recovery separately from this relative classification.

## Rationale

Nested groups directly test whether individually causal sites combine into a small sufficient necessity set or whether additional components continue to add recovery. Carrying forward the Day 8 order and controls prevents grouped results or held-out concepts from changing membership. Positive, negative, behavior, and block controls distinguish recovery from generic score movement or model disruption.

The all-layer-11 group answers the project brief's complete relevant-block question within the candidate vocabulary. It is not equivalent to replacing the entire residual block because it omits residual inputs and other internal tensors.

## Consequences

- Day 9 may estimate grouped rescue only; triggered-to-normal grouped sufficiency remains Day 10.
- The K=16 point is not a sum of Day 8 individual effects and must be measured directly.
- Validation may assess transfer but may not reorder or resize the prefixes.
- The distribution label is determined by the frozen relative 80% rule, while absolute completeness remains a separate result.
- Safety remains locked and no Day 9 action authorizes safety access.
