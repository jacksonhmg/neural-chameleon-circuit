# Day 30: Build and Audit the Private Candidate

## Phase information

- **Sprint phase:** Day 30
- **Status:** Complete
- **Calendar dates:** 2026-08-08

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
- Built the manuscript twice from source commit `92fbe3326c6875e9dca8536ba6be550dd35394a7` and compared the complete output bytes.
- Rendered all 14 pages at 144 DPI and inspected the title page, typography, margins, figures, captions, tables, references, appendices, page numbers, and glyph rendering.
- Ran the frozen numerical audit, citation/figure/provenance checks, and the complete 62-test suite through the candidate audit.
- Recorded acceptance as a local private candidate under Decision 0026.

#### Results and evidence

- Final private PDF: `output/pdf/neural-chameleon-causal-mechanisms-private.pdf`.
- PDF properties: 14 US-letter pages; six main figures; three appendix tables; 13 cited primary-source references.
- PDF SHA-256: `f5497295937645c76670d8a967099f6530bf131b08a4723607600bbb44cce1e9`.
- Determinism: two clean builds from the candidate source commit produced identical PDF bytes.
- Numerical audit: all 21 manuscript claims matched their sealed sources at the frozen tolerance.
- Visual audit: all 14 pages passed inspection with no clipping, overlap, broken glyphs, or illegible figures/tables.
- Candidate audit: 16/16 checks passed in `paper/audits/day30-private-candidate-audit.json`.
- Regression suite: `PYTHONPATH=src:scripts pytest -q tests` passed all 62 tests.
- External-state check: no push, tag, release, submission, upload, message, author contact, or external evaluation occurred.

#### Failures and unexpected observations

- A plain `pytest -q tests` run failed import collection because the available pytest environment did not have the repository installed.
- Adding `PYTHONPATH=src` resolved library imports but not the test's direct import of a Day 24 script. The complete verified command is `PYTHONPATH=src:scripts pytest -q tests`; it passes all 62 tests. No code or scientific result was changed.
- The first deterministic wrapper attempt passed an absolute auxiliary-file path to BibTeX. TeX's `openout_any=p` policy refused to write the `.blg` file there, so no final PDF was copied. The wrapper now enters the isolated temporary directory for BibTeX while using `BIBINPUTS` to read the bibliography from `paper/`. Manuscript content is unchanged.
- The next two complete builds had identical metadata and page-content stream hashes but different raw PDF hashes. Inspection isolated the difference to pdfTeX's random trailer `/ID`. The source now suppresses that non-semantic identifier with `\pdftrailerid{}`, allowing byte-for-byte deterministic builds.
- The first candidate-audit run reported 15/16 checks because the audit treated BibTeX's successful `warning$ -- 0` built-in-function count as an emitted warning. The diagnostic matcher was narrowed to actual BibTeX warning/error lines, after which all 16 checks passed. No PDF, citation, numerical value, or scientific result changed.

#### Artifacts and commits

- `scripts/day30_build_private_manuscript.sh`
- `scripts/day30_audit_private_manuscript.py`
- `output/pdf/neural-chameleon-causal-mechanisms-private.pdf`
- `paper/audits/day30-manuscript-build.log`
- `paper/audits/day30-manuscript-bibtex.log`
- `paper/audits/day30-visual-audit.json`
- `paper/audits/day30-private-candidate-audit.json`

## Open questions and blockers

- None.

## Decisions

- Decision 0026 accepts the audited PDF and its source package as the private manuscript candidate. It does not expand any scientific claim or authorize external action.

## Handoff

- **Current state:** The deterministic private manuscript candidate is complete and locally accepted.
- **Next action:** Conduct a private author-level content review and decide authorship, venue, and whether any further experiment is warranted before separately considering release authorization.
- **Verification still needed:** None for the Days 26–30 manuscript sprint. Independent replication and external review remain absent and are stated limitations, not unfinished Day 30 checks.
