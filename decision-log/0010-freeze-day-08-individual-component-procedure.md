# Decision 0010: Freeze the Day 8 Individual-Component Procedure

- Date: 2026-08-05
- Status: Accepted before the Day 8 scan

## Decision

Use the Day 04 candidate universe without modification: all 16 pre-output-projection query heads and the whole MLP at each frozen zero-based layer 12, 11, 10, and 9, for 68 candidates. Patch every non-padding response token. Individual head patches replace the selected 256-wide slice immediately before that layer's attention output projection; MLP patches replace the complete post-feedforward-layernorm branch before its residual addition.

On all 256 discovery-positive examples, calculate four inexpensive screens for every candidate:

1. response-token RMS of the normal-minus-triggered activation difference;
2. signed projection of that difference onto the normalized block-12 probe direction, mapping head differences through the relevant columns of the attention output projection first;
3. first-order attribution patching, the gradient of the triggered probe score dotted with the normal-minus-triggered activation difference; and
4. response-token RMS gradient sensitivity.

The screens are diagnostic only. Rank candidates exclusively by exact normal-to-triggered rescue on all discovery positives, using the Day 04 equal-concept macro recovery, shared-candidate gate, and tie-breaks. Require bit-exact complete/truncated agreement and same-condition identity patches for all 68 candidates in both conditions. Deterministic CPU vectorized-versus-independent tests must be exact. On the real BF16/MPS checkpoint, where expanding batch shape changes numerical kernels, require maximum absolute probe-score disagreement no greater than 0.002, fixed-continuation NLL disagreement no greater than 0.02, and use no more than two candidates per vector batch.

Freeze the first 16 gated candidates. Set final K to the largest available member of the already-frozen nested sizes 1, 2, 4, 8, and 16; therefore K is 16 when at least 16 candidates pass the gate. Select 16 distinct random controls outside the frozen top 16 by ascending SHA-256 of the string 42 followed by the candidate ID. Commit the ordered set, K, control identities, discovery summary hash, and component-set hash before loading individual-component validation results.

After that commit:

- run exact induction for the 16 selected and 16 random-control candidates on all discovery positives;
- run exact rescue and induction for the same 32 frozen identities on every validation positive, without reranking;
- compare selected candidates against the complete nonselected same-layer discovery distribution as an additional same-layer control; and
- measure fixed-continuation response NLL for both patch directions for the 16 selected candidates on a deterministic diagnostic subset containing two positive and two negative examples per benign concept, selected by ascending SHA-256 of day08-behavior plus the example ID.

Use 10,000 paired percentile-bootstrap replicates with seed 42. Resample complete example rows within concept and class, and weight concepts equally in macro results. Behavioral NLL is a bounded diagnostic, not a selection criterion or a complete behavioral-preservation claim.

## Rationale

The four screens cover magnitude, probe-direction alignment, first-order patch approximation, and gradient sensitivity, but none is itself a causal intervention. Exact rescue remains the only rank score, preserving the Day 04 preregistration.

The two-stage commit prevents held-out component effects from changing identities, order, K, or controls. Vectorizing independent candidates only changes batching. Same-shape identity checks remain exact, while the observable-scale BF16 tolerance records the already-known batch-shape numerical effect instead of mislabeling it as causal signal.

The behavior subset keeps full 42-block likelihood evaluation tractable while spanning every benign concept and both labels. It can identify obvious fixed-continuation damage but cannot establish generation quality or semantic preservation.

## Consequences

- Day 8 will produce an exact 68-candidate discovery leaderboard even if a screening method performs poorly.
- Validation may evaluate only the committed selected and random-control identities.
- Candidate-level head results refer to pre-output-projection slices, not isolated residual-stream contributions.
- Day 9 may test nested grouped sets but may not reorder the frozen candidates using validation or safety data.
- Safety remains locked. This decision and the later component-selection commit are prerequisites, not authorization to access safety examples.
