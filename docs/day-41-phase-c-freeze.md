# Phase C Semantic-Conditioning Freeze

Phase C tests whether the acquired K12 direct-reconfiguration effect is semantically controlled upstream. It uses existing data only and remains development evidence. It cannot earn the target title or count as fresh confirmation.

## Frozen population and conditions

Select eight positive examples per concept from all 13 concepts by a fixed SHA-256 ordering. Each correct concept is paired with a different concept whose rendered trigger has the same tokenizer length on every selected prompt. Teacher forcing preserves response IDs and masks.

For each example, evaluate normal, correct-trigger, Day 4 irrelevant-trigger, matched different-trigger, and hidden different-concept substitution conditions. The irrelevant trigger remains a separate specificity control even when its prompt length differs; it is not treated as the matched natural-text benchmark.

## Upstream intervention

The boundary is `layer_08.block_output`, strictly before the earliest selected layer-9 head. Replace only the tokenizer-aligned named-concept prompt rows in the correct-trigger residual with the corresponding different-trigger rows. Every other prompt and response row remains unchanged. This v1 contract forbids searching another boundary after outcomes.

Natural different-trigger and hidden-substitution changes are compared in four frozen spaces:

- the pooled selected-head K12 pattern;
- the direct-path complete standardized 13-probe vector;
- the total full-model complete probe vector; and
- the discovery-selected layer-11 F3 frontier vector.

The full K12 selected-head replacement is the central operation. The weaker layer-10 `concept_span_ov` result is retained only as a secondary source diagnostic.

## Upstream prediction

Predict the natural different-trigger direct-effect vector from a fixed 64-dimensional Rademacher projection of the layer-8 different-minus-correct concept-span representation plus a separate 64-dimensional projection of the normal response state. Compare with a normal-state-only ridge baseline under strict leave-one-concept-out evaluation. The projection seed, ridge coefficient, feature standardization, target, and folds are frozen.

## Continue gate

Phase C passes only if the full predictor beats its baseline, hidden substitution recreates at least half of the natural direct, total, frontier, and K12 patterns under one-sided bootstrap bounds, the irrelevant-control ratio is at most 0.75, and the hidden direct effect has the expected sign in at least nine concepts.

Passing permits drafting a separately committed fresh-confirmation contract and disjoint data-role manifest. Failure stops the program at the Phase A–B acquired direct-reconfiguration result. Exact clauses, IDs, pairs, hashes, expected rows, and prohibited adjustments are in [`frozen-phase-c-contract.json`](../results/day-41/frozen-phase-c-contract.json).
