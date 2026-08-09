# Days 31–35: Acquired-Writer Development Method

## Purpose and evidence status

This phase implements Section 6 of the frozen successor plan. It asks whether the prospectively frozen K12 candidate is an acquired writer computation and whether its observed trajectory contains held-out information about the layer-12 displacement and probe effects.

All examples and released probes in this phase have already been inspected by the completed project. Every result is therefore **mechanism development evidence**, never fresh confirmation. Passing this phase permits development of the pre-K12 predictor; it cannot by itself authorize the target title.

The authoritative machine-readable contract is [`results/day-31/frozen-acquired-writer-plan.json`](../results/day-31/frozen-acquired-writer-plan.json). If prose and JSON differ, the JSON controls.

## Frozen populations

Component resolution and realized-forward accounting cover all 1,732 records in the Day 4 discovery, validation, and safety corpora, including both labels. This prevents the mechanism description from silently discarding negative or inconvenient examples.

Acquisition and intermediate-state prediction use positive examples because the operational phenomenon is defined on concept-positive responses:

- fit all normalizers, bases, and diagnostic parameters on the 256 discovery positives;
- select ridge regularization by leave-one-discovery-concept-out validation; and
- evaluate once on all 610 positives from the seven validation concepts and two safety concepts.

Every held-out concept and both labels remain reportable separately. The evaluation population is held out only relative to the Gate 1 diagnostic fit; it is not fresh evidence for the successor project.

## Writer coordinate

`P_Z` is the identity map over the ordered K12 pre-`W_O` head slices and response-relative tokens. It has no fitted rotation, truncation, or outcome-selected subspace. Each head is divided by its discovery-Chameleon natural-delta RMS, with the frozen numerical floor in the contract. Padding is excluded and examples receive equal aggregate weight regardless of response length.

The identity choice makes the acquisition comparison deliberately strict: an apparent difference cannot be created by choosing a Chameleon-favoring low-rank projection after seeing precursor trajectories.

## Exact realized-forward accounting

For Gemma-2 block `l`, the captured additive branches are the tensors actually added to the residual:

```text
a_l = post_attention_layernorm(o_proj(concat_h y_lh))
m_l = post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x_l + a_l)))
x_(l+1) = x_l + a_l + m_l.
```

The layer-12 monitor state is reconstructed from the embedding residual and every realized `a_l` and `m_l` through block 12. The reconstruction is scored both in hidden space and after applying every released layer-12 probe.

Gemma-2's post-attention RMSNorm couples heads through one denominator. A head's accounting allocation is therefore:

```text
a_lh = gamma * o_proj_h(y_lh) / sqrt(mean(o_proj(concat_h y_lh)^2) + epsilon),
```

using the realized full-branch denominator. These allocations sum to the realized normalized attention branch up to declared floating-point tolerance. They are numerical accounting terms, not causal effects.

The committed-code real-checkpoint preflight showed that fused BF16 `W_O` and normalization kernels are not exactly decomposable into separately evaluated head slices. [Decision 0029](../decision-log/0029-freeze-bf16-accounting-clarification.md) therefore freezes symmetric allocation of the captured fused-projection and normalization/cast numerical residuals across heads. Both residual magnitudes are reported explicitly; the final allocation still closes to the actually realized normalized branch.

Any example exceeding a frozen closure tolerance invalidates the affected analysis. Results may not be averaged past a closure failure.

## Total and direct-path intervention definitions

The total effect patches a selected source state into a target run at the actual pre-`W_O` head hook, or patches a selected whole-MLP output, and permits ordinary downstream recomputation.

The direct-path effect freezes later additive writes as follows:

1. Cache every target-condition raw joint-head state, normalized attention branch, and normalized MLP branch through block 12.
2. At an intervened attention layer, replace only the selected raw head slices in the cached target joint-head tensor, then recompute that layer's `o_proj` and post-attention RMSNorm with their actual parameters and shared denominator.
3. Install that counterfactual normalized attention branch directly. Unselected raw head slices remain at their cached target values.
4. At an intervened MLP layer, install the selected source condition's already normalized MLP additive branch.
5. Freeze every nonintervened additive branch strictly downstream of the earliest intervention to its cached target value. At later intervened layers, install only the counterfactual branch defined above.

The residual stream still carries installed branch changes forward, but no unfrozen later additive write can amplify them. The downstream-dependent remainder is the total effect minus this direct-path effect under this exact operator. It is not described as a natural indirect effect.

To remove BF16 matrix-shape artifacts, the response-only attention recomputation is implemented as a matched delta anchored to the cached realized target branch: recompute both target and counterfactual response-only branches with identical kernels, subtract them, and add the difference to the cached target branch. This clarification makes identity exact without changing the selected-head counterfactual.

## Acquisition tests

Representation acquisition compares each example's precursor and Chameleon `T_c-N` K12 trajectories in the frozen identity coordinate. The primary aligned-amplitude coefficient is the projection of the precursor trajectory onto its paired Chameleon trajectory. A materially weaker precursor requires both the frozen point-estimate and upper-confidence-bound criteria.

Functional acquisition applies each checkpoint's observed natural K12 delta to its own cached normal trajectory under the same K12 clamp and evaluates all released probes at the same layer-12 site. The primary ratio compares the precursor's absolute macro raw-margin effect with the Chameleon's. Scores and causal rescue are secondary checks.

Both representation and functional clauses must pass. A parameter-distance comparison cannot substitute for either.

## Intermediate-state diagnostic

The full diagnostic uses the observed K12 delta, permitted normal `resid_post[8]` response state, and fixed response-position features to predict `u = x12(T_c)-x12(N)`. Discovery-only PCA bases reduce each head, the normal state, and the output; all targets and final metrics remain in the original layer-12 coordinate.

The comparison set is fixed before implementation:

- discovery training-set mean;
- leave-one-example-out concept-label mean, which is intentionally a strong outcome-informed baseline within each held-out concept;
- normal-state-only ridge;
- the identical model using the frozen layer-count-matched nonselected heads; and
- the identical model using exact-precursor trajectories.

The K12 diagnostic must improve both original-space `R2_u` and the complete released-probe margin-vector error. Correlation and cosine are secondary only.

## Gate and fallback

Gate 1 passes only when:

1. all accounting and intervention correctness checks pass;
2. both representation and functional acquisition satisfy their frozen material-weakness thresholds; and
3. the K12 diagnostic clears the absolute and best-baseline residual-prediction gates and the best-baseline probe-vector gate.

Failure of any scientific clause stops the successor title-level mechanism program. The repository then documents the strongest defensible causal-localization result and does not begin the concept-to-writer gate. A failed software or numerical check blocks interpretation until corrected; it is not counted as scientific evidence either way.

## Dissemination boundary

This phase permits local artifacts and commits only. It does not authorize push, tag, release, submission, upload, external messaging, or author contact.
