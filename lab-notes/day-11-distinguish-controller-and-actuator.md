# Day 11: Distinguish the Controller from the Actuator

## Phase information

- **Sprint phase:** Day 11
- **Status:** Complete
- **Calendar dates:** 2026-08-06
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

- Froze `day11-v1` before results, including the 88-example positive subset, five prompt conditions, six-token-region partition, 215 interventions per example, bootstrap procedure, evidence rules, numerical controls, and safety lock.
- Implemented exact tokenizer-offset source masks and eager-attention contribution capture for every query head in layers 0–12.
- Implemented causal pre-`o_proj` source-contribution rescue and induction, direct component-output rescue and induction, vectorized execution, deterministic resumption, analysis, visualization, and an independent verifier.
- Added four Day 11 tests covering exact source partitioning, contribution reconstruction and identity, frozen-grid expansion, and deterministic synthetic evidence classification; all 45 repository tests pass.
- Ran the pinned checkpoint over all 88 frozen examples and produced 19,360 unique rows: five baselines and 215 interventions per example.
- Generated equal-concept 10,000-replicate bootstrap summaries, seven analysis tables, three PNG/PDF figure pairs, and a hashed artifact manifest.
- Re-ran the full analysis and reproduced every generated pre-audit artifact byte for byte.
- Visually inspected all three 4,800 by 2,700 PNG figures and corrected a display-only rounding/color issue in the mechanism diagram.

#### Results and evidence

- All four frozen stage rules pass in discovery and validation. The official classification is `supported_controller_actuator_decomposition`.
- Selected attention monitoring-language rescue is small but positive: 0.00062 [0.00042, 0.00086] on discovery and 0.00365 [0.00202, 0.00555] on validation. The selected-minus-random intervals are also strictly positive.
- Named-concept rescue is 0.309 [0.289, 0.336] on discovery and 0.108 [0.099, 0.116] on validation, with positive selected-minus-random contrasts in both scopes.
- Prior-response-source rescue is 0.165 [0.148, 0.184] on discovery and 0.135 [0.121, 0.149] on validation, again exceeding the matched random group in both scopes.
- All-head causal source scans first have positive monitoring intervals at layers 1/6, named-concept intervals at layers 1/3, and response-source intervals at layer 6/6 for discovery/validation. Monitoring peaks at layer 6, named concept at layers 10/8, and response at layer 11 in both scopes.
- Discovery-dominant selected-head roles divide into five trigger-other, three named-concept, and four response-source heads. Eleven of the 12 discovery-frozen roles have a positive interval for the same region in validation; layer-11 head 14 is the exception.
- The selected four-MLP actuator directly rescues 0.645 [0.599, 0.687] and induces 0.525 [0.488, 0.561] on discovery; validation effects are 0.370 [0.339, 0.397] and 0.433 [0.410, 0.456]. All four required intervals are strictly positive.
- The direct selected-attention group carries large bidirectional effects, and selected K16 reaches 0.933 rescue/0.921 induction on discovery and 0.833/0.801 on validation. The block-12 positive control is exactly 1.0 in both directions and scopes.
- Prompt counterfactuals are diagnostic: concept-only wording produces 0.937 discovery and 0.902 validation suppression fractions, whereas monitoring-only produces 0.052 and 0.200. These wording-dependent effects are not the primary causal evidence.
- The complete result package passes all 25 independent audit checks.

#### Failures and unexpected observations

- Six separately evaluated BF16 attention-region matmuls produced a 0.0602 reconstruction discrepancy on the first real-checkpoint preflight. Float32 contribution decomposition reduced this to 0.0311. Assigning the remaining one-time BF16 rounding closure term to the always-present, non-hypothesis-critical template channel reduced the measured maximum reconstruction error to 0.00000191 without changing monitoring, named-concept, or response contributions.
- Repeated early-stopped MPS forwards retain memory until process exit. Full batches therefore reached the MPS ceiling late in the intervention grid. Execution was changed to committed, process-bounded two-example batches with exact key-level resumption. Several long pairs completed their source grid in one process and their remaining direct checks in the next; the final archive contains one result per frozen key and one implementation commit.
- The first mechanism figure rounded the supported monitoring effects to `0.00` and colored a descriptive prompt-context link green. The display now uses three decimals for source effects and gray for descriptive-only links.

## Open questions and blockers

- Monitoring-language contributions are causally detectable but very small in the selected group; Day 11 does not establish whether a different wording or component set carries most monitoring recognition.
- Source-channel onset identifies the earliest measured causal effect at the tested layer granularity, not the layer where semantic meaning is first computed.
- The high concept-only counterfactual effect may depend on the explicit wording `The named concept is ...`; broader prompt paraphrases remain untested.
- The positive-only teacher-forced scope does not establish negative-example selectivity, free-generation behavior, semantic preservation, or safety preservation.

## Decisions

- The complete Day 11 source-contribution and controller–actuator procedure is frozen before results in [Decision 0014](../decision-log/0014-freeze-day-11-controller-actuator-procedure.md).

## Handoff

- **Current state:** Day 11 is complete. The frozen controller–actuator decomposition is supported, all artifacts and figures are checked in, the 25-check audit passes, and safety remains locked.
- **Next action:** Begin Day 12 by freezing the benign cross-concept transfer analysis and final safety-test metrics before any new results or safety access.
- **Verification still needed:** None for the Day 11 completion gate.
