# Day 24: Coupled Autoregressive Generation

Day 24 tests the final mappings during actual greedy generation without reference-token leakage. After the unpatched first response token, every subsequent step recomputes normal and triggered donor activations from that intervention lane's exact generated prefix, applies the frozen destination-relative delta, and generates the next destination token.

## Outcome

The exact 80-row grid is sealed and all ten audit checks pass. It covers four deterministically frozen examples, normal and correctly triggered baselines, the K12 identity benchmark in both directions, and selected/null versions of all four mappings.

- Selected mappings exactly match their destination-base generation in 56.25% of lanes, versus 90.63% for route-matched nulls.
- Mean selected-to-base token F1 is 0.887, versus 0.967 for nulls.
- Deception rescue outputs are unchanged across every selected and null mapping in this subset. Deception induction changes one of two outputs for each selected within-layer mapping; selected cross-layer and null mappings remain unchanged.
- Harmfulness shows more greedy sensitivity. Selected induction mapping exact-match rates range from 0 to 0.5, and selected rescue rates from 0 to 0.5. Route-matched nulls range from 0.5 to 1.0.
- The locally blinded inspection found coherent, relevant outputs with no stance reversal: every harmfulness-positive output rejected or qualified the racial stereotype, and all benign fruit outputs remained benign and on-task.

These are four-example descriptive results. They show that selected routes can change realized output more often than null routes, particularly for harmfulness, while providing no observed evidence of gross degeneration or safety-stance reversal. They cannot establish general semantic equivalence.

## Files

- `coupled-generation-results.jsonl.gz`: sealed 80-row raw archive, including local generated text.
- `coupled-generation-metrics.csv`: condition-level automatic diagnostics.
- `coupled-generation-summary.json`: selected/null aggregate description.
- `qualitative-review-packet.md`: locally blinded output packet.
- `qualitative-review-key.json`: condition key, kept separate from the packet.
- `qualitative-rating-template.csv`: retained for a possible future independent rating; currently blank.
- `qualitative-review-assessment.md`: explicitly non-independent local review.
- `coupled-generation-preflight.json` and `day24-audit.json`: no-outcome preflight and final audit.

