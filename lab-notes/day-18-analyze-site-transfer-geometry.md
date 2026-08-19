# Day 18: Analyze Site-Transfer Geometry

## Phase information

- **Sprint phase:** Day 18
- **Status:** Complete
- **Calendar date:** 2026-08-07
- **Frozen protocol:** [Site-shuffling plan](../results/day-15/frozen-site-shuffling-plan.json)

## Objective

Determine whether portable effects are better explained in raw head coordinates or after each head's output projection into the shared residual stream.

## Session log

### 2026-08-07 — Session 1

#### Planned focus

- Capture benign trigger deltas for 12 selected and 12 matched-null heads.
- Compute exact least-squares output-projection alignment maps.
- Compare raw, per-example RMS-matched, and projection-aligned transport for all selected-head pairs.
- Quantify cosine, CKA, low-rank structure, and prediction of held-out safety-atlas outcomes.

#### Work performed

- Implemented a resumable benign geometry evaluator using the verified cached-tail runner.
- Passed the committed transform preflight: all 144 least-squares transforms are finite, the maximum self-map identity error is 0.00000262 against a 0.01 tolerance, no hooks remained, and the safety split was not accessed.
- Captured raw-head and output-projected trigger deltas for 12 selected and 12 matched-null heads on 64 positive discovery examples.
- Completed all raw, per-example RMS-matched, and output-projection-aligned rescue and induction interventions for the 144 ordered selected-head pairs.
- Sealed the 55,424-row archive, computed raw/residual cosine and linear CKA, ran the frozen 32-component PCA, fit the benign-only ridge predictor, and evaluated its association with the held-out Day 17 safety atlas.
- Generated and visually inspected both geometry figures; no layout defect required correction.

#### Results and evidence

- The audit passes all six checks: exact and unique raw rows, expected geometry shapes, finite metrics, complete 132-pair held-out comparison, and figures present.
- Raw mean-delta cosine is the strongest single benign-transfer correlate (Pearson 0.612). Output-projection cosine is 0.390 and residual-space cosine is only 0.101.
- The nine-feature benign model has Pearson/Spearman 0.866/0.248 on benign pair magnitudes. Its held-out association with safety delta transfer is 0.695/0.181. The gap between linear and rank correlation means a few large routes are predictable, while global route ranking is weak.
- Reciprocal `layer_11.head_08`/`layer_11.head_09` transport again dominates. Benign raw rescue/induction are 0.1239/0.0864 in one direction and 0.1802/0.1502 in the other.
- Projection alignment produces no broad rescue improvement (mean nonidentity 0.00877 versus 0.00839 raw) and only a modest induction increase (0.00553 versus 0.00431 raw).
- Residual delta variance is moderately distributed: 14.8% in PC1, 60.2% in the first 8, 77.4% in the first 16, and 91.0% in the first 32.
- Evidence: [result package](../results/day-18/README.md), [summary](../results/day-18/geometry-summary.json), [audit](../results/day-18/geometry-audit.json), and [prediction table](../results/day-18/geometry-transfer-prediction.csv).

#### Failures and unexpected observations

- The initial two-example run was interrupted after 7,244 rows to test whether four-example batches improved throughput. Four-example batching completed another 1,836 rows without a memory failure but was slower per row, so it was stopped and the exact row-key archive resumed under the original two-example setting. This operational trial changed no intervention, estimand, or result; the final audit requires all 55,424 unique rows.
- Output-projection alignment was less helpful than the portable-code account might predict. It did not make arbitrary routes reliable or recover the extreme raw effects of the best reciprocal layer-11 pair.

## Decisions

- No new durable decision; coordinate tests were frozen in Day 15.

## Handoff

- **Current state:** Day 18 complete; geometry, transport, prediction, and audit packages sealed.
- **Next action:** Run Day 19's frozen permutation ensemble and benign-only confirmation-mapping selection.
- **Verification still needed:** None for the Day 18 completion gate.
