# Day 6: Localize the Residual Stream

## Phase information

- **Sprint phase:** Day 6
- **Status:** Complete
- **Calendar dates:** 2026-08-04
- **Frozen input contract:** [`day04-v1`](../data/splits/day04-v1/README.md)

## Objective

Use bidirectional `resid_post` patches to localize the released monitor-evasion effect across all 42 blocks and three token regions—aligned prompt, response, and their union—using only the four discovery concepts' positive examples. Produce rescue and induction heatmaps, quantify paired uncertainty, and freeze the top four coarse layers under the Day 4 rule.

## Session log

### 2026-08-04 — Session 1

#### Planned focus

- Specify and test semantic prompt-token alignment before using prompt patches.
- Implement a deterministic, resumable layer-by-region evaluator.
- Validate early termination at the measured monitor and same-condition identity patches.
- Run all discovery-positive rescue and induction interventions without loading validation or safety data.
- Analyze layer/concept effects, freeze the top four full-response rescue layers, render heatmaps, and audit the package.

#### Work performed

- Froze the exact discovery-only patch procedure in Decision 0007 before inspecting intervention results, including semantic prompt alignment, response-relative alignment, all-aligned union, early termination, structural post-monitor nulls, uncertainty, onset, and ranking.
- Added reusable exact prompt-alignment, aligned residual patching, truncated monitor execution, and paired localization analysis utilities with six new focused tests.
- Confirmed on the real checkpoint that stopping after `resid_post[12]` produces bit-identical probe scores to a complete 42-block forward.
- Ran 78 real-checkpoint identity controls: normal-to-normal and triggered-to-triggered for every executable layer and token region. Every score difference was exactly zero.
- Evaluated all 256 positive examples from the four discovery concepts. For each example, recorded two baselines and all 42 layers × three token regions × two causal directions, yielding 65,024 rows.
- Computed per-concept and equal-concept recovery and induction fractions with the frozen 10,000-replicate paired bootstrap.
- Applied the Day 4 full-response rescue ranking without modification and froze layers `[12, 11, 10, 9]` in Decision 0008.
- Rendered separate rescue and induction heatmaps and ran the independent 20-check package audit.

#### Results and evidence

- The complete result and interpretation are in [`results/day-06`](../results/day-06/README.md); all 20 independent audit checks pass.
- Macro response rescue for layers 8–12 is 0.0771, 0.2161, 0.5359, 0.9721, and 1.0000. Macro induction is 0.0870, 0.2852, 0.6485, 0.9663, and 1.0000.
- The top-four ranking is unambiguous: layers 12, 11, 10, and 9, with the fifth-ranked layer 8 substantially lower.
- Layer 11 is consistently strong across concepts: response rescue ranges from 0.9338 for German to 1.0029 for comforting, while induction ranges from 0.9192 for HTML to 0.9897 for comforting.
- Layer 10 is more heterogeneous: response rescue is 0.9707 for comforting but only 0.1045 for German. Day 7 must therefore test whether attention/MLP mediation is shared or concept-dependent inside the frozen layer range.
- Prompt-only macro effects remain small, peaking at 0.0160 rescue and 0.0682 induction. All-aligned induction exceeds response-only induction by 0.1725 at layer 9 and 0.1220 at layer 10, indicating an interaction between aligned prompt state and response computation.
- Statistically detectable effects begin earlier than substantively large effects. The profile supports distributed buildup with strong late concentration across layers 9–11.
- Validation was never loaded and the safety split remains locked and unscored.

#### Failures and unexpected observations

- No model run, identity control, or raw-record assertion failed. The resumable evaluator completed in one process.
- Length sorting exposed large runtime variation: the longest prompt/response batches took several times longer than the early batches. Row-level flushing and key-based resumption kept this operational rather than methodological.
- The first rendered heatmaps placed the structural-null label too close to the title. Visual QA caught the overlap; moving the label inside the shaded post-monitor region fixed it without changing data or scales.
- The inferential-onset criterion flags effects as early as layers 0–1, but those estimates are below 0.3%. This was not used to change the frozen ranking and is documented to prevent statistical significance from being mistaken for mechanistic concentration.

## Open questions and blockers

- No Day 6 blocker remains. Day 7 must determine whether attention, MLP, or both mediate the late-layer effect and whether the pattern transfers without reranking.

## Decisions

- The exact execution procedure is recorded before the scan in [Decision 0007](../decision-log/0007-freeze-day-06-localization-procedure.md).
- The frozen top-four layer set is `[12, 11, 10, 9]`: [Decision 0008](../decision-log/0008-retain-layers-09-through-12.md).

## Handoff

- **Current state:** Day 6 is complete. Layers 12, 11, 10, and 9 are frozen for subsequent component analysis; validation and safety remain unopened by this phase.
- **Next action:** On Day 7, compare full attention output, full MLP output, and complete block output at the four frozen layers. Establish the strongest component type on discovery, then test held-out benign examples without changing the layers or reranking per concept.
- **Carry forward:** Preserve exact response pairing, token-region definitions, identity controls, irrelevant-trigger/random-layer controls, paired uncertainty, and the discovery-only selection boundary. Keep safety locked.

## Day closeout

- **Outcome:** Complete. Every checklist item, completion gate, and heatmap deliverable is satisfied.
- **Verification:** 20/20 package-audit checks pass; 78/78 identity controls are exact; full/truncated execution is bit-identical; all 23 automated tests pass; artifacts regenerate deterministically.
- **Commits:** Procedure freeze commit `51bb609`; results closeout commit containing the full audited Day 6 package.
