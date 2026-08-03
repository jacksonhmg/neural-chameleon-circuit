# Day 4 Experimental Split and Analysis Freeze

This document is the human-readable preregistration for freeze `day04-v1`. The machine-readable contract is [`analysis-plan.json`](../data/splits/day04-v1/analysis-plan.json), and the exact file hashes and counts are in the [`manifest`](../data/splits/day04-v1/manifest.json). Both were frozen before any component search began.

## Concept roles

| Role | Concepts | Permitted use |
|---|---|---|
| Discovery | HTML, biology-focused, comforting, German | Baselines, localization, component ranking, ordered top-16 set, and final `K` |
| Held-out validation | all-caps, chemistry-based, confused, Finnish, jokey, literature-focused, mathematical | Transfer evaluation with the discovery-frozen identities and `K`; no reranking |
| Locked safety | deception and harmfulness | One confirmatory evaluation after explicit unlock; no safety-specific selection |

“Held out” for the seven benign concepts means held out from this project's component discovery and selection. It does not mean the released organism or benign probes never saw those concepts: all 11 benign concepts were part of the authors' training setup. Cross-concept evidence therefore tests whether a circuit selected on four concepts transfers to seven other trained benign concepts.

## Benign split construction

The source is the pinned released topical-response dataset `mega_dataset_evaluated_20250729_144844_harm_batch_train.json` at minimal-reproduction revision `f077304f6bebd31ded4cc868f2115a8c81067960`, SHA-256 `f731099c0c46f4f55670233eadde0092b15a69cfd039d37d5a8ac55b6c882f43`.

For target concept `c`:

- a positive example must have been generated for `c`, must not be a topical refusal, and must have topical rating `rating(c) ≥ 0.75`;
- a negative example must have been generated for a different concept in the same analysis role, must not be a topical refusal, and must have `rating(c) ≤ 0.25`.

These generation-category and rating rules match the authors' probe-training data construction. Negative donors are restricted to the same role, so no validation-concept source response appears in discovery. Each concept receives exactly 64 examples per class. Candidate records are ordered by salted SHA-256, not by dataset order or observed probe score. Source rows and prompt/response content are globally unique across the 1,408 benign examples.

## Safety split construction

- **Harmfulness:** all 100 positive and 100 negative released records from `harmful_samples_simple.json` are retained without score-based filtering.
- **Deception:** reproduce the released RepE loader: keep true-labeled facts (306 available, below the 512 cap), use the first 80% as the released train portion, and retain the final 20%—62 statements—as test responses. Each response is paired with the released deceptive context as positive and honest context as negative. The response text is identical within a pair. Deception's label is the assistant context, not the truth value of the statement.

The safety file was materialized and structurally verified, but no model, probe, activation, or output score was computed. The fail-closed access rule is documented in [`SAFETY_LOCK.md`](../data/splits/day04-v1/SAFETY_LOCK.md).

## Frozen primary measurements

- **Example probe score:** masked arithmetic mean of response-token sigmoid probabilities, as specified on Day 2.
- **Classification:** per-concept AUROC is primary; also report TPR at 5% FPR. The latter uses the normal-condition negative-score empirical 95th percentile with the higher order statistic, and classifies scores strictly greater than that threshold as positive.
- **Suppression gap:** `mean(normal positive) - mean(correct-trigger positive)`.
- **Causal rescue:** the unclipped recovery fraction already fixed in `EXPERIMENT_SPEC.md`, computed as a ratio of positive-example dataset means.
- **Causal induction:** the corresponding unclipped reverse-direction fraction.
- **Behavior preservation:** paired difference in mean response-token negative log likelihood on the exact same teacher-forced response IDs.

Negative examples determine classification performance and false activation/damage controls. Positive examples define the suppression, recovery, and induction denominators because the causal question is recovery of the monitor signal where the monitored concept is present. Every ratio must be accompanied by its raw means, numerator, and denominator. If the baseline suppression denominator is not positive, the recovery estimate is unstable and cannot enter component ranking.

Correct and irrelevant triggers are fixed per concept in `analysis-plan.json`. Irrelevant triggers cycle within discovery or validation roles; both safety probes use German as the irrelevant trigger. Every paired condition reuses identical response-token IDs.

## Frozen uncertainty procedure

Use 10,000 paired nonparametric percentile-bootstrap replicates with seed 42 and 95% intervals. Resample complete source-example rows so every prompt condition and intervention remains paired. Classification resampling is stratified by concept and class. For macro results, resample within each concept independently, compute the concept statistic, and take an equal-weight concept mean.

## Frozen component-selection rule

1. On discovery data only, scan all 42 blocks with full-response `resid_post` normal-to-triggered rescue patches and rank blocks by the equal-weight macro mean of the four concept recovery fractions.
2. Retain the top four zero-based blocks. At each, make all 16 pre-`o_proj` query heads and the whole MLP eligible, for 68 individual candidates.
3. Exact-patch every candidate on discovery positives. Its primary rank score is the equal-weight mean recovery across the four concepts.
4. A shared candidate must have finite positive recovery on at least three of four discovery concepts.
5. Sort by macro recovery, then minimum concept recovery, lower layer, attention head before MLP, and lower head index.
6. Freeze the first 16 eligible components, or every eligible component if fewer than 16 qualify. Test nested prefixes `K ∈ {1, 2, 4, 8, 16}` where available, and choose the final reported `K` using discovery only before validation is opened.
7. Apply the same ordered identities and final `K` to every validation and safety concept without reranking.

Required controls are same-condition identity patches, the full measured `resid_post[12]` positive control, and deterministic random eligible-component sets outside the frozen top-16 order for each reported `K`. Any nonzero identity-patch score or logit difference invalidates the affected patch result and must be resolved before interpretation.

## Verification

[`split-audit.json`](../results/day-04/split-audit.json) records 15 passing checks for byte-for-byte regeneration, manifest integrity, counts, labels, disjointness, provenance, content hashes, deception pairing, absence of result fields, and the fail-closed safety loader. The audit intentionally performs no model or probe evaluation.
