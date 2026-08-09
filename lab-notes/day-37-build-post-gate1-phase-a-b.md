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

## Handoff

- **Current state:** The complete Phase A–B implementation is synthetically verified and ready for its committed real-checkpoint preflight.
- **Next action:** Commit this implementation boundary, run and inspect only the two-checkpoint preflight, then begin population execution only if it passes.
- **Mandatory stop:** Do not begin the scientific matrix until every implementation check and real-checkpoint preflight passes.
