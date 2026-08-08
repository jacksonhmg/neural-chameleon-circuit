# Day 30: Build and Audit the Private Candidate

## Phase information

- **Sprint phase:** Day 30
- **Status:** In progress
- **Calendar dates:** 2026-08-08
- **Sprint tracker:** [SPRINT.md](../SPRINT.md)

## Objective

Produce a deterministic private manuscript PDF and audit its scientific traceability, build integrity, visual layout, provenance, and no-release boundary.

## Session log

### 2026-08-08 — Session 1

#### Planned focus

- Commit the build and audit machinery, build twice, inspect every page, and seal the local candidate.

#### Work performed

- Added a deterministic LaTeX/BibTeX build using the source commit timestamp through `SOURCE_DATE_EPOCH`.
- Added a package audit for PDF structure/text, warnings, citations, numerical claims, figures, tests, Git provenance, and release status.
- Ignored ephemeral LaTeX build intermediates while reserving `output/pdf/` for the final local PDF.

#### Results and evidence

- Pending implementation commit, deterministic build, visual inspection, and package audit.

#### Failures and unexpected observations

- A plain `pytest -q tests` run failed import collection because the available pytest environment did not have the repository installed.
- Adding `PYTHONPATH=src` resolved library imports but not the test's direct import of a Day 24 script. The complete verified command is `PYTHONPATH=src:scripts pytest -q tests`; it passes all 62 tests. No code or scientific result was changed.
- The first deterministic wrapper attempt passed an absolute auxiliary-file path to BibTeX. TeX's `openout_any=p` policy refused to write the `.blg` file there, so no final PDF was copied. The wrapper now enters the isolated temporary directory for BibTeX while using `BIBINPUTS` to read the bibliography from `paper/`. Manuscript content is unchanged.

#### Artifacts and commits

- `scripts/day30_build_private_manuscript.sh`
- `scripts/day30_audit_private_manuscript.py`

## Open questions and blockers

- None.

## Decisions

- No new scientific decision. The output remains a private candidate under Decision 0025.

## Handoff

- **Current state:** Build and audit machinery implemented.
- **Next action:** Commit machinery, build the PDF deterministically, and inspect all pages.
- **Verification still needed:** All Day 30 completion-gate checks.
