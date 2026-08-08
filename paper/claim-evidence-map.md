# Manuscript Claim–Evidence Map

This file is the editorial audit surface for the manuscript. The full scientific ledger remains [`report/claim-ledger.md`](../report/claim-ledger.md).

| Paper ID | Permitted claim | Evidence class | Direct evidence | Required qualification |
| --- | --- | --- | --- | --- |
| P1 | The released checkpoint reproduces trigger-dependent activation-probe suppression across benign and safety concepts. | Reproduction / baseline | Days 1 and 5; Day 13 safety baselines | Phenomenon only; not causal evidence |
| P2 | A benign-selected K16 population causally rescues and induces evasion on locked deception and harmfulness examples. | **Primary confirmatory** | `results/day-13/safety-transfer-metrics.csv`; clean reproduction | One checkpoint and released probe family |
| P3 | The effect is concentrated in late response-token computation and is head-dominated. | Held-out mechanistic plus post-confirmatory falsification | Days 6–10; Day 14 decomposition | K16 is not minimal; MLP K4 is near zero on the falsification subset |
| P4 | The same selected population transfers across held-out benign concepts and two locked safety concepts. | Held-out benign plus primary confirmatory | Days 12 and 13 | “Shared” refers to the tested concepts and unchanged selected population |
| P5 | Trigger-linked activation information can transfer through a fixed population of non-original routes. | Prospective intervention | Day 21 confirmation | Population-average, sparse, mapping-sensitive; one cross-layer map reverses |
| P6 | Final-map transport is behaviorally selective rather than output-inert or broadly behavior-driving. | Prospective behavioral endpoints | Day 23, with Day 24 descriptive generation | Deception output coupling is small; harmfulness within-layer coupling is stronger; four-example generation is descriptive |
| P7 | The supported mechanism is a downstream, head-dominated causal population or subspace, not a uniquely routed minimal mixed circuit. | Synthesis | P2–P6 and Day 14 falsification | “Circuit” is operational and non-unique |

## Numerical anchors

| Quantity | Value used in main text | Source |
| --- | --- | --- |
| Deception K16 rescue / induction | 0.947 / 0.987 | Day 13 safety transfer |
| Harmfulness K16 rescue / induction | 0.844 / 0.726 | Day 13 safety transfer |
| Day 23 deception directional output coefficient | 0.022 induction; 0.028 rescue | Day 23 four-map population |
| Day 23 harmfulness directional output coefficient | 0.087 induction; 0.098 rescue | Day 23 four-map population |
| Coupled generation exact match | 56.25% selected; 90.63% route-matched null | Day 24 descriptive grid |
| Coupled generation token F1 | 0.887 selected; 0.967 route-matched null | Day 24 descriptive grid |

Intervals must accompany these values in the Results or caption when the source supplies them. Rounded prose values do not replace the source table.

## Forbidden upgrades

- “the mechanism” → use “a causally influential population” unless uniqueness is explicitly disclaimed;
- “replicated” → use “reproduced from a detached clean worktree” for the same checkpoint and data;
- “arbitrary routes” → use “a fixed, mapping-sensitive population of non-original routes”;
- “no behavioral effect” → use “bounded under the frozen teacher-forced metrics”;
- “general monitor-evasion circuit” → use “single-checkpoint causal case study.”
