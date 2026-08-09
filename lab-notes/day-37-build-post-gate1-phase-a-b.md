# Day 37: Build and Execute Post–Gate 1 Phase A–B

## Phase information

- **Sprint phase:** Day 37
- **Status:** In progress
- **Calendar date:** 2026-08-09
- **Frozen contract:** [Phase A–B contract](../results/day-36/frozen-phase-a-b-contract.json)
- **Execution authorization:** [Decision 0033](../decision-log/0033-authorize-post-gate1-phase-a-b-execution.md)

## Objective

Implement, verify, execute, and mechanically adjudicate the frozen Phase A variance/carrier diagnostics and Phase B causal-operation identification program.

## Session log

### 2026-08-09 — Session 1

#### Planned focus

- Record the explicit execution authorization before implementation.
- Build deterministic Phase A centering, variance, residualized, and outer-concept diagnostic machinery.
- Build Phase B zero/mean/replacement, rank-matched random, downstream-frontier, and attention-operation machinery.
- Add synthetic tests and final-shape real-checkpoint preflights before scientific execution.
- Execute only committed code under one persisted execution identity.

#### Evidence status

No Phase A–B scientific outcome has been produced at this authorization boundary.

#### Phase A implementation checkpoint

- Added `post_gate1_diagnostics.py` with the exact response-decile rule, discovery-global fallbacks, leave-one-example-out same-concept centering, variance decomposition, residualized ridge diagnostics, and fully outer-folded cross-concept prediction.
- The outer-concept implementation refits every selected-head, nonselected-head, precursor-head, normal-state, and target-residual PCA inside the outer fold. Inner leave-one-training-concept-out selection uses only probe-vector SNMSE, and semantic features come from the exact precursor input embedding table.
- Added a committed-only reducer that consumes the sealed Gate 1 feature tensors and emits exactly the four required Phase A artifacts. It rechecks pinned input hashes, population counts, checkpoint token identity, execution identity, and the sealed probe standardization before reduction.
- Added seven focused tests covering the exact decile formula, feature validation, RMS normalization, example exclusion, singleton fallback, variance decomposition, equal-example concept summaries, and outer-fold target exclusion.
- Verification: `7 passed`; focused Ruff and Python compilation pass.
- No real Phase A reducer was run and no scientific outcome was inspected. This preserves the implementation-gate boundary while Phase B remains unfinished.

#### Prospective row-accounting correction

- The synthetic execution-matrix test found that all three attention formulas in the frozen contract contain an unsupported factor of two in their listed result.
- Froze Decision 0034 and a machine-readable clarification before any outcome access. No scientific dimension changed; only the expected attention counts and grand total were corrected.
- Correct Phase B total: 444,224 effect rows. The implementation will retain both the parent mismatch and clarification hash in its audit.

#### Phase B operator checkpoint

- Implemented exact zero, natural source, frozen mean, and arbitrary raw pre-`o_proj` head replacement caches for ordinary total and sealed frozen-write direct paths.
- Implemented deterministic PCG64/QR Haar controls with positive-`R` diagonal and mandatory norm, temporal-Gram, and retained-singular-value checks before BF16 installation.
- Implemented the exact downstream branch order and all 32 unique `F0`, cumulative `Fj`, and unique single-branch `Bj` frontier cells across source layers 9–12. Frozen branches are the exact complement of the declared release set.
- Implemented eager-attention capture of post-softmax patterns, post-RoPE Q/K, V, and raw query-head outputs, plus pattern-with-target-values, aligned-value-with-target-pattern, concept-span QK removal/reverse-installation, and concept-span OV removal/reverse-installation operators.
- Added ten focused Phase B tests, including exact replacement, Haar invariants, frontier complement, declared-head/source isolation, bidirectional concept operation, 16-site expansion, and corrected 444,224-row arithmetic. Focused Ruff and the relevant 12-test causal-mechanism suite pass.
- No real-checkpoint preflight or population outcome has yet been run.

#### Complete implementation-gate checkpoint

- Added the resumable Phase B runner with separate mean-capture, absolute/random, frontier-discovery, attention-discovery, deterministic selection, frontier-evaluation, and attention-evaluation stages. Every stage shares one persisted execution identity and refuses uncommitted code.
- Added the deterministic reducer for hierarchical 10,000-replicate bootstrap bounds, absolute signatures, direct/downstream fractions, complete-vector recovery, frontier recovery, matched controls, sign/safety stability, exact row/key audits, manifests, and the mechanical continue gate.
- Added a two-checkpoint final-batch preflight covering realized accounting, identity replacement, cached/full equality, all replacement families, Haar invariants, frontier complements, and all four attention operators.
- Froze Decision 0035 and the exact attention alignment/removal/reverse-installation conventions before checkpoint execution.
- Focused Phase A–B verification: 29 tests pass. Complete project suite with `PYTHONPATH=src:scripts`: 96 tests pass. Focused Ruff and Python compilation pass.
- A repository-root pytest invocation also collected the ignored external research checkout and failed on its uninstalled `jaxtyping` dependency; the project-owned `tests/` suite is complete and passing.
- No Phase A reducer, real-checkpoint preflight, population intervention, discovery selection, or scientific gate has been run at this checkpoint.

#### Real-checkpoint preflight attempt 1 — failed

- Commit: `6f80b2e`.
- Chameleon accounting, raw-head identity, absolute operator shapes, Haar invariants, cached/full replay, and frontier complement construction completed before the attention capture.
- Attention capture failed because the `with_kwargs` hook read an empty positional tuple while Gemma supplied `hidden_states` by keyword.
- No attention effect, precursor preflight, Phase A outcome, population intervention, or selection score was produced.
- Preserved the failed attempt and Decision 0036. The only code change reads keyword `hidden_states` first and retains the positional fallback.

#### Real-checkpoint preflight attempt 2 — passed

- Commit: `9b6a14e`.
- Both Chameleon and exact precursor passed with batch size two and response width 123.
- Raw-head identity total/direct errors and cached/full margin/score errors were exactly zero; Haar audits passed; all four attention operations returned `[2, 123, 3584]`; response IDs/masks matched; zero hooks remained.

#### Phase A attempt 1 — startup validation failure

- The reducer stopped before loading an input because it checked the shorthand status `frozen` instead of the contract's exact `frozen-before-post-gate1-phase-a-b-outcomes`.
- No tensor or scientific outcome was accessed. Preserved the attempt and Decision 0037.
- Fixed only the exact status checks in both population runners. The complete preflight must be rerun under the new commit before Phase A restarts.

#### Preflight attempt 3 and Phase A completion

- The complete two-checkpoint preflight passed again under `d0b6c5d` with the same zero identity/cached errors, passing Haar audit, four attention shapes, response alignment, and hook cleanup.
- Phase A then completed on all 866 positives. Its implementation/leakage audit passed and reproduced the sealed probe standardization exactly.

#### Mean capture attempt 1 — superseded before effects

- Both checkpoint-specific normal donor sweeps completed on all 1,732 records under `d0b6c5d`; no Phase B effect row was produced.
- Before starting effects, code review found that the random-control path would recompute identical frozen QR matrices and large Gram/SVD diagnostics for every batch and draw.
- Decision 0038 caches the 384 seed-defined matrices and uses their float64 orthogonality residual plus a matrix-product rounding term as a uniform certificate for norm, temporal-Gram, and retained-singular-value preservation.
- The completed attempt-1 donor tables remain local with attempt suffixes. A new commit, full preflight, and donor regeneration are required before effects.

#### Final donor attempt 2 and effect attempt 2 — performance restart

- Preflight passed under `fcc63f6`; both donor tables then regenerated on all 1,732 records and matched attempt 1 with maximum aggregate-sum error exactly zero.
- The effect matrix began with job chunk size four. Count-only monitoring stopped it after 168 natural, 1,008 absolute, and 2,304 random rows; no value was inspected.
- Measured throughput projected roughly 3.5 hours for random controls alone.
- Decision 0039 preserves all attempt rows, forbids mixing them into the final matrix, caches target branch recomputations per direction, and requires a new two-checkpoint preflight with sixteen expanded jobs before restarting effects from zero.

#### Final donor attempt 3 and effect attempt 3 — direct-path performance restart

- The larger sixteen-job total-effect kernel passed the complete two-checkpoint preflight under `9986735`; both donor tables regenerated on all 1,732 records and matched the previous two attempts with maximum aggregate-sum error exactly zero.
- Count-only monitoring of the restarted effect matrix stopped it after 148 natural, 888 absolute, and 1,152 random rows. No scientific value was inspected.
- Total-effect throughput improved, but the direct path still projected each intervention sequentially and dominated runtime.
- Decision 0040 preserves all attempt rows, forbids mixing them into the final matrix, batches algebraically independent direct paths in candidate-major batch rows, and requires exact equality to sixteen independent direct executions on both checkpoints before another population restart.

#### Batched-direct preflight attempt 4 — failed

- The expanded-batch direct implementation differed from independent BF16 direct execution by maximum absolute hidden-state error `0.015625` in both checkpoints. This violates the frozen exact-equality rule for cache/kernel optimizations, despite lying within the general accounting tolerance.
- No donor was regenerated, no population was accessed, and no scientific value was inspected under this commit.
- A timing-only Chameleon preflight benchmark found the batched implementation slower than 32 cached independent projections: 0.923 versus 0.581 seconds. Probe summarization was only 0.016 seconds.
- Decision 0041 rejects the batched-direct output entirely, restores independent direct execution, and prospectively raises only the total-effect chunk to the natural 32-random-draw family size. A new two-checkpoint exact-equality preflight is mandatory.

#### Preflight attempt 5 passed; effect attempt 4 exposed the patch-hook bottleneck

- The 32-job total-effect kernel passed both checkpoints with exact cached/full margin and score equality, zero identity error, and no leaked hooks.
- Donor attempt 4 completed all 1,732 records and 280 cells in both checkpoints. Counts and aggregate sums matched donor attempts 1–3 exactly.
- Count-only monitoring stopped the effect matrix after 152 natural, 912 absolute, and 1,408 random rows; no scientific value was inspected.
- Code audit found that the vectorized hook cloned the full expanded activation once for every patched job/member, up to 384 redundant clones per random-control call.
- Decision 0042 freezes exact old-hook preflight result hashes before replacing those repeated clones with one clone plus the same ordered in-place assignments. Both checkpoint hashes must reproduce exactly before another population restart.

#### Preflight attempt 6 passed; effect attempt 5 isolated per-site transfer cost

- The one-clone hook reproduced the frozen Chameleon and precursor result hashes exactly, with zero cached/full errors and no leaked hooks.
- Donor attempt 5 again matched every count and aggregate sum from attempts 1–4 exactly.
- The first positive cell showed no meaningful throughput improvement. Count-only monitoring stopped at 144 natural, 864 absolute, and 896 random rows; no scientific value was inspected.
- The interrupt stack located active time in per-member mask/source transfers: up to 384 calls for one 32-draw selected-head intervention.
- Decision 0043 batches disjoint expanded-batch rows by exact patch site and retains the same old-kernel hashes as a mandatory two-checkpoint exact-equality gate.

#### Preflight attempt 7 passed; effect attempt 6 isolated redundant prefix execution

- Site-batched transfers reproduced both frozen old-kernel hashes exactly; cached/full errors remained zero and hook cleanup passed.
- Donor attempt 6 again matched every count and aggregate sum from attempts 1–5 exactly.
- Count-only monitoring stopped the population run at 512 natural, 3,072 absolute, and 16,128 random rows; no scientific value was inspected.
- Cached-tail execution was still computing the identical layers 0–8 prefix on every repeated job row. For 32 jobs at batch size two, that meant 64 redundant prefix rows before the four live layers.
- Decision 0044 captures the two-example layer-9 input once and repeats that cache across jobs. Exact full/cached equality and both frozen old-kernel hashes remain mandatory before another restart.

## Handoff

- **Current state:** The complete Phase A–B implementation is synthetically verified and ready for its committed real-checkpoint preflight.
- **Next action:** Commit this implementation boundary, run and inspect only the two-checkpoint preflight, then begin population execution only if it passes.
- **Mandatory stop:** Do not begin the scientific matrix until every implementation check and real-checkpoint preflight passes.
