# Day 27: Build Manuscript Figures

## Phase information

- **Sprint phase:** Day 27
- **Status:** Complete
- **Calendar dates:** 2026-08-08

## Objective

Generate six deterministic, self-contained main-text figures with source data, adverse results, and a hash manifest.

## Session log

### 2026-08-08 — Session 1

#### Planned focus

- Implement the frozen six-figure plan and verify both raster and vector outputs.

#### Work performed

- Added a single manuscript-figure builder that reads only existing sealed result artifacts.
- Added deterministic PDF metadata and machine-readable per-figure source tables.
- Preserved the K12/K4 decomposition, maximum matched nulls, reversed cross-layer mapping, and harmfulness route-level output coupling in the main panels.
- Rebuilt every output twice and compared SHA-256 hashes; all PNG, PDF, CSV, and manifest hashes were stable.
- Rendered every PDF through Poppler and verified single-page integrity, dimensions, typography, legends, labels, error bars, and absence of clipping or overlap.

#### Results and evidence

- All six figures generated successfully with six source-data tables and a single hash manifest.
- Initial PNG inspection found all panels legible except a descriptive note in Figure 6C that overlapped the bars. The note was removed and its scope retained in the panel title and manuscript caption.

#### Failures and unexpected observations

- Figure 6C's first layout placed a scope note behind the bars. No data or interpretation changed; the corrected panel communicates the four-example boundary in its title.

#### Artifacts and commits

- `scripts/day27_build_manuscript_figures.py`

## Open questions and blockers

- None.

## Decisions

- No new durable decision; the builder implements Decision 0025 and the frozen figure plan.

## Handoff

- **Current state:** Six figures, six source tables, and the manifest are complete and visually verified.
- **Next action:** Draft Methods and Results around these frozen figures.
- **Verification still needed:** Manuscript-level numeric and cross-reference audit.

## Day closeout

- **Outcome:** Complete
- **Completed on:** 2026-08-08
- **Final evidence:** `paper/figures/figure-manifest.json` and `paper/source-data/figure-01.csv` through `figure-06.csv`.
- **Final commits:** Recorded after commit creation.
- **Carry-forward items:** Figure captions and numerical traceability move to Day 28.
