# Post–Gate 1 Phase A–B Implementation

## Authority and evidence boundary

The implementation follows the [frozen Phase A–B contract](../results/day-36/frozen-phase-a-b-contract.json), the [row-arithmetic clarification](../results/day-37/frozen-phase-b-row-accounting-clarification.json), and the [attention-operator implementation freeze](../results/day-37/frozen-attention-operator-implementation.json). The sealed original Gate 1 remains failed. All Phase A–B evidence is existing-data development evidence, never fresh confirmation.

No population reducer may run from uncommitted source. Every runner records one Git commit, execution identity, contract hash, and clarification hashes. Working JSONL files are resumable; deterministic summaries and their raw hashes are tracked at closeout.

## Phase A

`post_gate1_diagnostics.py` implements:

- the exact ten-bin response-position formula;
- discovery-global decile fallbacks;
- same-concept, same-decile leave-one-example-out centering with target-ID audits;
- full-residual and standardized complete-probe-vector variance decomposition;
- discovery-only PCA/ridge fitting for centered selected K12, normal-state, nonselected-head, and exact-precursor predictors; and
- thirteen fully outer-folded leave-one-concept-out predictions. Every PCA, target transform, alpha choice, and baseline is refit without the held-out concept.

The semantic baseline is the mean exact-precursor input-embedding vector for tokenizer tokens of the literal concept string with no special tokens. Phase A emits exactly the four contract artifacts under `results/day-38/` and explicitly retains the original uncentered failure.

## Phase B absolute and random operators

All K12 sites are 256-dimensional raw query-head slices before `o_proj`, restricted to valid response tokens. Zero, natural source, and leave-one-example-out normal concept/label/decile mean tensors are installed with target padding, dtype, and kernel geometry. Direct effects recompute the affected normalized attention branches with the actual shared RMS denominator and freeze every later additive write to the target cache.

Normal mean cells exclude the complete target example. Empty cells use the discovery same-label global normal decile mean. Both checkpoint-specific donor tables are constructed in a normal-only first pass.

Each random draw right-multiplies each natural token-by-256 head delta by its deterministic PCG64/QR Haar matrix. Norm, temporal Gram, and retained singular values must pass their pre-BF16 tolerances or execution stops.

## Downstream frontiers

The source branch is the named selected or same-layer nonselected head group. Later branches are ordered as source-layer MLP, then each later full attention and MLP branch through block 12. A release cell patches the source raw heads and freezes exactly the complement of its declared later branches.

The unique discovery counts are 14, 10, 6, and 2 configurations for source layers 9, 10, 11, and 12, respectively. Discovery selection uses only Chameleon discovery positives and the frozen equal-concept remainder-recovery tie rule. Heldout and negative runs label direct, selected, and total as three analysis roles even if two roles happen to execute the same frozen configuration.

## Attention operations

The runner captures eager post-softmax patterns, post-RoPE Q/K, V states, and raw query-head outputs. Pattern, value, concept-QK, and concept-OV operators follow Decision 0035. Unselected heads remain bitwise unchanged in the constructed raw tensor. Each operation is evaluated with ordinary downstream recomputation and the frozen direct path.

## Execution order

1. Run all synthetic tests.
2. Run `day37_verify_phase_b.py` on a final batch of two examples in both checkpoints.
3. Run the committed Phase A reducer once.
4. Run checkpoint-specific normal-mean capture.
5. Run absolute and random effects.
6. Run both discovery screens.
7. Generate the deterministic discovery selection manifest.
8. Run heldout-positive and all-negative frontier and attention matrices.
9. Run the deterministic reducer twice by byte comparison, audit every key/count/identity, and mechanically apply the gates.

The corrected Phase B total is 444,224 effect rows. Natural endpoints, mean tables, audits, and Phase A tensors are outside that count.

## Stop rule

An implementation failure blocks interpretation. A scientific continue-gate failure ends the mechanism program at the sealed causal-localization fallback. A pass permits drafting a new Phase C contract; it does not authorize Phase C execution.
