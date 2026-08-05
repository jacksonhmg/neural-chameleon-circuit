# Decision 0011: Freeze the Day 8 Individual-Component Set

- Date: 2026-08-05
- Status: Accepted after discovery and before individual-component validation

## Decision

Freeze the following discovery-ranked component order for all later exact tests:

1. `layer_11.mlp`
2. `layer_11.head_08`
3. `layer_10.head_12`
4. `layer_11.head_09`
5. `layer_12.mlp`
6. `layer_10.mlp`
7. `layer_10.head_02`
8. `layer_09.head_11`
9. `layer_12.head_12`
10. `layer_09.mlp`
11. `layer_12.head_03`
12. `layer_09.head_04`
13. `layer_11.head_14`
14. `layer_12.head_02`
15. `layer_09.head_13`
16. `layer_11.head_15`

Forty-two of 68 candidates passed the shared discovery gate, so the preregistered largest-available-prefix rule fixes final K at 16. The ordered set hash is `5ef5ab558b3383d0eb4d6cb6050e21fdf89b1c4d47b2e18287156ac09316e4d6`.

Freeze the 16 control identities in `results/day-08/frozen-component-selection.json`, selected outside the top 16 by the preregistered ascending SHA-256 rule. Use exactly these selected and control identities for discovery induction and held-out rescue and induction. Do not rerank, replace, or resize the set from validation results.

## Evidence

The discovery scan contains the complete 17,664-row grid: 256 discovery-positive examples, one destination baseline and 68 independent candidate patches per example. The discovery audit passed all 18 checks. Deterministic regeneration reproduced every tracked archive, summary, table, figure, selection file, and manifest byte for byte.

The top discovery effects were `layer_11.mlp` at 0.297 equal-concept macro rescue (95% paired-bootstrap interval 0.286–0.308) and `layer_11.head_08` at 0.249 (0.239–0.259). The exact ranking and uncertainty are recorded in `results/day-08/discovery-candidate-summary.json`; neither validation nor safety data contributed to selection.

## Rationale

This commit is the information barrier required by Decision 0010 and the Day 04 freeze. Persisting the identities, order, K, controls, hashes, and discovery evidence before the held-out run makes it possible to distinguish genuine transfer from validation-driven component selection.

## Consequences

- Day 8 confirmation must use the committed K=16 selected set and its 16 committed controls unchanged.
- Day 9 grouped tests inherit this order for the frozen nested prefixes 1, 2, 4, 8, and 16.
- Validation may measure transfer but may not alter the component set.
- Safety remains locked; this selection does not authorize safety access.
