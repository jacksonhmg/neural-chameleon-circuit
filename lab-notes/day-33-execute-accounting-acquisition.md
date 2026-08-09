# Day 33: Execute Complete-Corpus Accounting and Acquisition

## Phase information

- **Sprint phase:** Day 33
- **Status:** Complete
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
- Stopped and archived a second early population attempt after measuring the conservative eight-group chunk throughput. The final execution contract now hashes and persists the launch commit, batch size, group chunk size, and population limit; every row and feature tensor carries that execution ID. The real-checkpoint preflight now exercises all 27 frozen groups in one job-major batch before authorizing the higher-throughput shape.
- The 27-group real-checkpoint preflight passed in one expanded batch with exact equality (`0.0` maximum difference) for all 13 probes and monitor RMS, and no live hooks. This authorizes `group_chunk_size=27` with the separately frozen `batch_size=2` execution shape.
- Stopped and archived the final negative-only prefix before any positive feature artifact was written. The feature serializer now retains writer and monitor displacements as float32 differences of the realized BF16 endpoints instead of rounding those differences back to BF16; the normal realized residual remains BF16. The reducer enforces this dtype contract.
- A timing audit isolated separately launched direct-path branch recomputations as the remaining avoidable overhead. Stopped and archived the eight complete negative batches, then batched the 27 independent matched `W_O`/RMS branch deltas by layer while retaining BF16 residual-addition order. Synthetic equivalence is exact; the real-checkpoint preflight must also pass exactly before population execution resumes.
- The vectorized direct-path real-checkpoint preflight failed: its maximum hidden difference from the frozen independent operator was `0.03125`, above the existing `0.02` numerical tolerance. The failed artifact is preserved and the optimization was removed. Population execution therefore uses the independently evaluated, previously verified direct-path operator.
- Stopped and archived a final short negative-only prefix to improve throughput without changing numerical operators. The candidate execution shape uses `batch_size=4` and group chunks `13,13,1`, keeping the largest expanded batch at 52 versus the already verified 54 while halving independent direct-path launches per example. It remains blocked until exact full-versus-cached agreement passes on four real examples for every chunk shape.
- The four-example preflight passed every `13,13,1` group chunk with exact equality (`0.0` maximum difference) for all probe margins, sequence scores, and monitor RMS, with no live hooks. This is the final authorized population execution shape.
- A final timing trace showed that every independent direct group redundantly recomputed the identical target-side response-only `W_O`/RMS branch. Stopped and archived four complete negative batches, then cached that target tensor once per layer while leaving every counterfactual recomputation and its `[batch,response,hidden]` BF16 kernel shape unchanged. Synthetic cached-versus-uncached output is exact; real-checkpoint exactness is required before reuse.
- The first real cache preflight stopped before comparison because its validator incorrectly expected the 4096-dimensional concatenated query-head width after `W_O`; the actual realized additive branch is 3584-dimensional. Corrected the assertion to the captured target-branch shape. No scientific result artifact was produced by this software failure.
- The corrected real-checkpoint preflight passed: cached and uncached direct-path monitors were exactly equal (`0.0` maximum hidden difference) for one head at each layer, K12, and K16. Total cached-tail results also remained exact for all `13,13,1` chunks. This authorizes target-side direct-branch reuse without changing any counterfactual kernel shape.

### 2026-08-09 — Session 2

#### Work performed

- Executed the frozen full population with `batch_size=4` and group chunks `13,13,1` from commit `b0292e2a98d575827f82570985edcfbdfa2d583a`.
- Completed all 1,732 Chameleon records and all 866 paired positive examples from the exact precursor under execution ID `ba510c5c95b26836342ad635e4700fdd0cdac0b71f31939ee7f8691b14adc399`.
- Persisted and audited 1,300 accounting rows, 187,056 component-effect rows, 5,196 natural-endpoint rows, 866 precursor-functional rows, and 866 feature tensors for each checkpoint.
- Ran the committed deterministic reducer twice from the same analysis commit and compared all principal output hashes.

#### Results and evidence

- Every implementation check passed, including exact row/key coverage, feature dtypes and IDs, committed provenance, real-checkpoint preflight identity, and input hashes.
- Realized-forward accounting maxima were `0.0` hidden error, `0.0000076294` attention-allocation error, `0.0` probe-margin error, and `0.0` sequence-score error.
- Exact-precursor representation acquisition passed: absolute macro aligned amplitude `0.0112973`, one-sided 95% UCB `0.0159774`.
- Exact-precursor functional acquisition passed: precursor/Chameleon effect ratio `0.0324234`, one-sided 95% UCB `0.0512249`.
- On positive examples, K12 reached `0.94999` of K16's absolute induction effect and `0.86816` of K16's absolute rescue effect. These are causal-localization comparisons, not an acquired-writer conclusion.
- The five reducer outputs were byte-identical across both runs. Their hashes are recorded in the [result README](../results/day-33/README.md).

#### Failures and unexpected observations

- No full-run numerical or software failure occurred.
- The acquisition result passed but does not independently satisfy Gate 1; the observed-trajectory diagnostic remained required and was run on Day 34.

#### Artifacts and commits

- [Day 33–35 result package](../results/day-33/README.md)
- [Execution artifact manifest](../results/day-33/execution-artifact-manifest.json)
- Result commit: `2073cb5` (`record acquired writer gate 1 result`)

## Handoff

- **Current state:** Full accounting, component resolution, and exact-precursor acquisition are complete and audited.
- **Next action:** Apply the observed-trajectory diagnostic and Gate 1 disposition in Days 34–35.
- **Verification still needed:** Final repository closeout audit.

## Day closeout

- **Outcome:** Complete
- **Completed on:** 2026-08-09
- **Final evidence:** [Day 33–35 result package](../results/day-33/README.md), [`component-resolution-summary.json`](../results/day-33/component-resolution-summary.json), and [`acquired-writer-summary.json`](../results/day-33/acquired-writer-summary.json)
- **Final commits:** `2073cb5`; documentation closeout recorded in subsequent git history.
- **Carry-forward items:** The Day 34 diagnostic and Day 35 gate disposition, both now complete.
