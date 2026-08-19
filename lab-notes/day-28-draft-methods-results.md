# Day 28: Draft Methods and Results

## Phase information

- **Sprint phase:** Day 28
- **Status:** Complete
- **Calendar dates:** 2026-08-08

## Objective

Write a traceable Methods and Results draft around the six frozen figures, preserving every evidence-class boundary and material adverse result.

## Session log

### 2026-08-08 — Session 1

#### Work performed

- Drafted the experimental setting, monitor definition, split chronology, causal metrics, selection procedure, inference, site-transfer intervention, and behavioral endpoints.
- Drafted six Results subsections aligned one-to-one with the main figures.
- Added method and extended-result appendices covering artifact provenance, split counts, exact K16 identities, falsification, route gates, behavioral dispositions, and package integrity.
- Added a machine-readable numerical-claim ledger for the manuscript's central values.
- Added an independent audit script that resolves each ledger entry back to its sealed result row.

#### Results and evidence

- The draft reports all four frozen safety cells with intervals.
- The route section retains the failed cross-layer map and limits prospective language to the intervention.
- The behavioral section rejects exact output-inertness and keeps harmfulness rescue mechanically mixed.
- The 21-entry numerical ledger matches the sealed source rows exactly under a $5\times10^{-7}$ audit tolerance.

#### Failures and unexpected observations

- A secondary rounded-text anchor check initially searched only `paper/manuscript.tex` for the `3,880` Day 23 row count. That value correctly appears in `paper/appendix/extended-results.tex`; the check scope, not the manuscript, was wrong. The source-value audit passed without modification.

#### Artifacts and commits

- `paper/manuscript.tex`
- `paper/appendix/method-details.tex`
- `paper/appendix/extended-results.tex`
- `paper/numerical-claims.csv`

## Open questions and blockers

- Framing, citations, discussion, and limitations remain for Day 29.

## Decisions

- No new durable decision; prose implements Decision 0025.

## Handoff

- **Current state:** Methods and Results draft and numerical audit are complete.
- **Next action:** Write framing and verified primary-source references.
- **Verification still needed:** Citation resolution and final LaTeX build.

## Day closeout

- **Outcome:** Complete
- **Completed on:** 2026-08-08
- **Final evidence:** `paper/manuscript.tex`, appendices, `paper/numerical-claims.csv`, and `paper/audits/day28-numerical-audit.json`.
- **Final commits:** Recorded after commit creation.
- **Carry-forward items:** Framing, bibliography, discussion, and limitations move to Day 29.
