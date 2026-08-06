# Day 11: Distinguish the Controller from the Actuator

## Phase information

- **Sprint phase:** Day 11
- **Status:** In progress
- **Calendar dates:** 2026-08-06 to present
- **Sprint tracker:** [SPRINT.md](../SPRINT.md)
- **Frozen component order:** [Day 8 selection](../results/day-08/frozen-component-selection.json)
- **Sufficiency handoff:** [Day 10 results](../results/day-10/README.md)

## Objective

Test whether monitor evasion separates into causal stages that recognize monitoring language, identify the named concept, process the prompt and response through selected attention heads, and apply the downstream anti-probe transformation through selected MLP and complete component outputs.

## Session log

### 2026-08-06 — Session 1

#### Planned focus

- Freeze the positive mechanistic subset, prompt counterfactuals, exact source-position contribution patch, layer scan, selected/random groups, direct MLP tests, evidence rules, numerical controls, and safety isolation.
- Implement and test attention-contribution capture and vectorized region replacement.
- Execute the complete frozen grid on the real checkpoint.
- Produce source-region tables, a provisional causal mechanism diagram, complete provenance, and an independent audit.

#### Work performed

- Work in progress.

#### Results and evidence

- Work in progress.

#### Failures and unexpected observations

- None yet.

## Open questions and blockers

- Which layers and selected heads causally depend on monitoring-language tokens remains pending.
- Which depend on the named concept rather than generic trigger wording remains pending.
- Whether prompt-source effects and prior-response-source effects occupy different heads remains pending.
- Whether selected MLP response outputs form a supported downstream actuator remains pending.

## Decisions

- The complete Day 11 source-contribution and controller–actuator procedure is frozen before results in [Decision 0014](../decision-log/0014-freeze-day-11-controller-actuator-procedure.md).

## Handoff

- **Current state:** Day 11 procedure is frozen; no Day 11 model result exists and safety remains locked.
- **Next action:** Implement, test, and execute the 19,360-row controller–actuator grid.
- **Verification still needed:** Source-mask partitioning, contribution reconstruction, identity and vector equivalence, complete coverage, deterministic summary regeneration, evidence-rule reproduction, figure inspection, artifact hashes, and safety isolation.
