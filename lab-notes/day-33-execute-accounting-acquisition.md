# Day 33: Execute Complete-Corpus Accounting and Acquisition

## Phase information

- **Sprint phase:** Day 33
- **Status:** In progress
- **Calendar date:** 2026-08-09
- **Frozen protocol:** [Gate 1 contract](../results/day-31/frozen-acquired-writer-plan.json)
- **Verified machinery:** [Day 32 audit](../results/day-32/machinery-audit.json)

## Objective

Resolve K12 and its required comparison groups on every existing example, close realized-forward accounting, decompose total and frozen-write direct-path effects, and compare natural and functional K12 trajectories with the exact precursor.

## Session log

### 2026-08-09 — Session 1

#### Planned focus

- Add a resumable, hash-auditable execution pipeline.
- Vectorize independent intervention groups while retaining all 13 raw-margin and sequence-score endpoints.
- Store large positive-example trajectory tensors only in ignored local artifacts; commit deterministic summaries, hashes, and audits.
- Execute the Chameleon population first, then the exact precursor, without changing the frozen contract.

#### Work performed

- Began the vectorized all-probe execution runner and full-corpus script.
- Smoke accounting passed across all 13 concepts and both checkpoints. A four- then eight-job intervention smoke showed that expanded transformer cost remained approximately linear in jobs.
- Replaced redundant direct-path transformer executions with the algebraically identical replay of cached normalized additive branches required by the frozen-write operator; total effects still use actual intervened transformer forwards.
- Added exact cached-tail replay from natural `resid_pre[9]` for total effects. All selected sites are at layers 9–12, so blocks 0–8 are invariant cached prefix computation rather than part of an intervention job.
- Added a committed real-checkpoint equivalence preflight comparing full and cached-tail execution for an individual layer-9 head and complete K12 across all 13 probes.
- The first real-checkpoint preflight failed: reusing a prefix captured at base batch size changed BF16 results after job-major batch expansion (maximum raw-margin difference `0.0382315`). The failed output is preserved as `cached-tail-preflight-attempt-1.json`; the optimization remained blocked.
- Tightened the cache contract to capture `resid_pre[9]` only after exact job-major batch expansion. Prefixes are reused only between chunks with the same expanded batch shape and condition.
- The revised real-checkpoint preflight passed with exact equality (`0.0` maximum difference) for raw margins, sequence scores, and monitor activation RMS across an individual layer-9 head and K12, with all hooks removed.
- Archived all partial smoke data without deletion under ignored `artifacts/mechanism-gate1-v1-smoke-attempts/`. The tracked manifest records four row-file hashes/counts and both feature-tree manifest hashes. The complete frozen execution therefore starts with no reusable scientific rows or tensors from superseded implementations.
- Stopped and archived the first 14 accounting batches of the population attempt after noticing that launch provenance existed only in terminal output. No positive feature tensors had yet been written. Tightened the pipeline so every JSONL row and tensor contains the exact execution commit and every completed batch asserts that no hooks remain registered.

## Handoff

- **Current state:** Execution implementation in progress.
- **Next action:** Commit and smoke-test the resumable pipeline, then run the complete Chameleon and precursor populations.
