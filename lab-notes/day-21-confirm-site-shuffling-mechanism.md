# Day 21: Confirm the Site-Shuffling Mechanism

## Phase information

- **Sprint phase:** Day 21
- **Status:** Complete
- **Calendar date:** 2026-08-07
- **Frozen protocol:** [Site-shuffling plan](../results/day-15/frozen-site-shuffling-plan.json)

## Objective

Apply the prospectively frozen causal gate to four benign-selected mappings on every remaining positive safety example, with route-matched null sources and no safety-specific mapping selection.

## Session log

### 2026-08-07 — Session 1

#### Planned focus

- Commit the evaluator and then materialize and commit the authorization before any confirmation outcome.
- Pin all 130 example IDs, four selected maps, four route-matched null maps, and the exact 6,500-row grid.
- Run conditional and destination-relative delta rescue/induction, apply the fixed bootstrap gate, and reuse the pre-confirmation Day 20 negative controls by hash.
- Audit raw completeness, provenance, hooks, tests, figures, and documentation.

#### Work performed

- Implemented the authorization generator, resumable confirmation evaluator, independent gate analyzer, and unit checks.
- Committed the implementation at `67becda` before authorization.
- Materialized the authorization while all Day 21 outcome artifacts were absent. It pins 46 deception and 84 harmfulness examples, four selected and four route-matched null maps, 50 conditions per example, 6,500 expected rows, the bootstrap, and the mechanical gate.
- Committed the authorization at `78df5eb`, then ran the no-outcome model preflight. All six authorization, map, ID, grid, and hook checks passed; the preflight explicitly generated no confirmation outcome.
- The first outcome attempt stopped before any shuffled intervention because of the documented per-base assertion error. Corrected and committed the mechanical count, then resumed from four baseline rows.
- Completed and sealed all 6,500 authorized rows: 2,300 deception rows and 4,200 harmfulness rows.
- Applied the precommitted 10,000-replicate bootstrap and mechanical gate without manual score inspection first.
- Hash-verified and incorporated the eight pre-confirmation Day 20 negative specificity cells.
- Generated and visually inspected the primary confirmation figure, then passed the nine-check final audit.
- Ran the corrected complete-package audit. All 15 cross-day checks passed, including every daily audit/preflight, all eight sealed raw row counts, authorization-before-results Git ancestry, hashes, no resumable working files, all 57 tests, documentation, and the local-only/no-tag state.

#### Results and evidence

- The mechanical disposition is `portable_support`.
- All eight selected-source conditional/delta population estimates have positive lower 95% bounds, from harmfulness induction delta 0.0929 [0.0818, 0.1038] to deception induction delta 0.3867 [0.3619, 0.4103].
- All eight selected-minus-null contrasts also have positive lower bounds. Route-matched null population estimates are near zero or adverse in every primary cell.
- Mapping results are heterogeneous. The two within-layer maps are strong throughout; `cross_15130` is smaller but positive; `cross_15122` is significantly reversed in all eight primary cells.
- The four-mapping population was fixed before safety access, so the failed mapping remains included. It does not prevent the predeclared mean gate from passing, and it materially limits any interchangeability claim.
- Fixed negative-class means are 0.0470 for deception and −0.0367 for harmfulness across the four selected mappings.
- Secondary absolute effects are much larger than conditional/delta effects, and same-condition mismatch is nonzero. The corrected estimates and null contrasts show that mismatch inflates but does not fully explain transport.
- Evidence: [result package](../results/day-21/README.md), [gate](../results/day-21/confirmation-gate.json), [cell table](../results/day-21/confirmation-cells.csv), and [audit](../results/day-21/day21-audit.json).

#### Failures and unexpected observations

- The first confirmation attempt stopped before running any intervention because the runner's defensive assertion expected 48 jobs inside each base condition. The authorized grid specifies 48 interventions across two bases, so the correct per-base count is 24. Four baseline rows for the first two examples had already been written; no shuffled outcome was generated. The assertion was corrected without changing a map, source, destination, example, condition, estimand, or gate, and the resumable archive will retain those baseline keys.
- One benign-selected cross-layer mapping, `cross_15122`, failed consistently despite passing the benign eligibility rule. This is retained as evidence against homogeneous portability.
- The first complete-package audit passed every scientific audit, preflight, raw-row count, hash, Git-ancestry, test, and dissemination check but failed two documentation predicates. The sprint's Day 21 heading had accidentally retained `In progress`, and the checker expected the literal phrase `route-sensitive` where the report consistently used `route-dependent`/`layer-sensitive`. The tracker and checker literal were corrected; no scientific artifact or interpretation changed.

## Decisions

- [Decision 0021](../decision-log/0021-authorize-prospective-site-shuffling-confirmation.md) records the exact prospective authorization and prohibits safety-specific changes.
- [Decision 0022](../decision-log/0022-accept-qualified-portable-site-transfer.md) accepts the passing population gate while rejecting homogeneous or cleanly concept-specific wording.

## Handoff

- **Current state:** Days 15–21 complete; raw results, gate, negative controls, synthesis, figures, daily audits, and the passing complete-package audit are sealed.
- **Next action:** User review. Any release, push, author contact, or external transmission remains separately withheld.
- **Verification still needed:** None for the site-shuffling follow-up completion gate.
