# Day 20: Test Shuffle Specificity and Behavior

## Phase information

- **Sprint phase:** Day 20
- **Status:** Complete
- **Calendar date:** 2026-08-07
- **Frozen protocol:** [Site-shuffling plan](../results/day-15/frozen-site-shuffling-plan.json)

## Objective

Interpret portable probe effects alongside trigger specificity, class specificity, token order, cross-probe effects, activation-distribution distance, and full-model behavioral perturbation.

## Session log

### 2026-08-07 — Session 1

#### Planned focus

- Run source-condition, label, cross-probe, and response-token-order controls.
- Measure source/destination, delta/natural-delta, and patched/destination RMS ratios.
- Run full-model NLL and KL diagnostics for five frozen key interventions.

#### Work performed

- Implemented the resumable specificity and distributional evaluator.
- Passed the 64-example, six-mapping specificity preflight and sealed all 6,400 planned rows.
- Tested correct, irrelevant-concept, monitoring-only, and malformed sources with aligned and reversed response-token order on positive and negative examples.
- Scored both the target safety probe and the other safety probe for every intervention.
- Recorded source/destination, transported/natural-delta, and patched/destination RMS ratios.
- Ran and sealed 224 full-model NLL/KL rows for the five Day 15-frozen intervention jobs plus both baselines.
- Ran the 10,000-replicate analyzer, generated the diagnostic figure, visually inspected it, and passed the six-check audit.

#### Results and evidence

- Selected within-layer routes are strongly source-specific for deception: correct-trigger effects are 0.6265/0.8002, versus irrelevant-concept 0.0813/0.0232. Harmfulness is only partially source-specific: 0.1878/0.1705 correct versus 0.1385/0.1020 irrelevant.
- Monitoring-only and malformed source effects are near zero for both concepts.
- Selected within-layer effects are much smaller on negative target examples than positive examples.
- Cross-probe behavior is asymmetric. Deception-source deltas weakly affect the harmfulness probe, but harmfulness-source deltas strongly suppress the deception probe by 0.3118–0.3294 raw score. This exceeds the corresponding harmfulness-probe shifts of 0.0628–0.0692.
- Reversing response-token order barely reduces the aggregate effect, so exact tokenwise source alignment is not necessary.
- The best within routes have transported/natural-delta RMS ratios close to one, but patched activations remain elevated: patched/destination RMS is 1.32–1.38 for deception and 1.63–1.64 for harmfulness.
- Destination-relative pilot deltas have low full-model KL (0.00184–0.00206 deception; 0.00616–0.00835 harmfulness) compared with natural triggers (0.0652; 0.1617) and raw shuffled replacements (0.0153–0.0169; 0.0273–0.0356).
- Evidence: [result package](../results/day-20/README.md), [specificity metrics](../results/day-20/specificity-metrics.csv), [behavior metrics](../results/day-20/behavior-metrics.csv), and [audit](../results/day-20/day20-audit.json).

#### Failures and unexpected observations

- Harmfulness did not show the anticipated clean source or cross-probe specificity. Irrelevant-concept sources retained much of the harmfulness-probe effect, while correct harmfulness sources caused a larger shift in the deception probe than the target probe.
- Response-token reversal was unexpectedly ineffective as a negative control. This favors a token-order-insensitive shared shift or manifold perturbation over exact token-aligned message passing.
- Even the benign-selected within-layer destinations are 32–64% above their normal RMS after patching, so low KL cannot be interpreted as fully on-manifold activation replacement.

## Decisions

- No new durable decision; diagnostics were frozen on Day 15.

## Handoff

- **Current state:** Day 20 complete; specificity, activation-distance, behavior, figures, and audit are sealed.
- **Next action:** Commit the Day 21 confirmation authorization before reading any remaining safety outcomes, then run the frozen causal gate.
- **Verification still needed:** None for the Day 20 completion gate.
