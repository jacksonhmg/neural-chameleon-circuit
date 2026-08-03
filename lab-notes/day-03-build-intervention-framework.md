# Day 3: Build the Paired Intervention Framework

## Phase information

- **Sprint phase:** Day 3
- **Status:** Complete
- **Calendar dates:** 2026-08-02 to 2026-08-02
- **Sprint tracker:** [SPRINT.md](../SPRINT.md)
- **Experiment specification:** [`EXPERIMENT_SPEC.md`](../EXPERIMENT_SPEC.md)

## Objective

Implement and verify a paired teacher-forced forward-pass runner that can capture and patch the specified Gemma-2 activation locations in either direction while preserving exact response tokens and safely managing hooks.

## Session log

### 2026-08-02 — Session 1

#### Planned focus

- Resolve the exact Gemma-2 module boundaries for all required patch targets.
- Implement paired normal and monitoring-trigger conditions with shared response IDs.
- Implement activation capture and bidirectional patches for residual, attention, MLP, block, and individual-head targets.
- Add fast correctness tests and a full released-checkpoint integration test.

#### Work performed

- Audited the installed Transformers 4.57.1 Gemma-2 attention, MLP, and decoder-block forward paths and mapped each required intervention to a precise module input or output.
- Implemented `src/neural_chameleon/interventions.py` with typed patch sites, paired batch preparation, activation capture, normal-to-triggered rescue, triggered-to-normal induction, optional response-token selection, released-probe scoring, and the locked recovery/induction formulas.
- Tokenized each response exactly once, reused its ID/mask tensors across conditions, left-padded prompts, right-padded responses, and constructed padding-independent position IDs for batched runs.
- Made every activation cache carry the exact response IDs and mask. The runner rejects a mismatched continuation or mask before installing a hook.
- Implemented `resid_pre`, normalized attention-branch, normalized MLP-branch, full block-output/`resid_post`, and individual pre-`o_proj` query-head capture and patching.
- Registered patch hooks before capture hooks so captures at patched sites record the values actually sent downstream. Removed every hook in `finally`, including on forward failure.
- Added nine deterministic tests covering all targets, both directions, identity, success/failure cleanup, batch/single equivalence, token selection, token mismatch rejection, and metric sign conventions.
- Added and ran `scripts/day03_verify_runner.py` on the pinned released 9B checkpoint and German probe using two teacher-forced positive examples.
- Documented the public runner API, hook locations, shapes, safety invariants, commands, and current response-relative scope.

#### Results and evidence

- All nine deterministic tests passed in under one second after tokenizer setup.
- The released-checkpoint integration report completed with status `pass`; all 11 recorded correctness checks passed.
- Normal German probe scores were `0.44341168` and `0.26520747` (mean `0.35430956`). Triggered scores were `0.00007792` and `0.00012173` (mean `0.00009983`).
- A normal identity patch produced exactly zero score difference and zero response-logit difference.
- Patching the complete normal `resid_post[12]` into the triggered run exactly restored the normal activation and score, yielding recovery `1.0`.
- Patching the complete triggered `resid_post[12]` into the normal run exactly reproduced the triggered activation and score, yielding induction `1.0`.
- On the real Gemma model, all five required target types were captured at the expected shapes and reproduced their source values with maximum absolute patch error `0.0`.
- All hooks were removed after the runs; the remaining registered-hook count was `0`.
- Batched versus single German probe scores differed by at most `0.00074354`. Response-state relative L2 error was at most `0.7451%` with cosine at least `0.99997342`; response-logit relative L2 error was at most `1.5310%` with cosine at least `0.99979293`.
- The exact shared response-token IDs are recorded in the report with SHA-256 `9c7e8679dd50e5a2a2ae7c38439e3fd87b2ba5693709a8602c34e6ac3c721afe`.
- Full evidence: [Day 3 intervention-runner verification](../results/day-03/README.md).

#### Failures and unexpected observations

- The first released-checkpoint run failed only the batch/single activation check because an initial absolute tolerance of `0.01` was inappropriate for BF16 residual values reaching magnitude `94`; the maximum coordinate difference was `0.5`.
- A diagnostic rerun showed activation cosine above `0.99997`, activation relative L2 error below `0.75%`, logit cosine above `0.99979`, and probe-score error below `0.000744`. An intentionally strict `0.1%` relative-error criterion still failed.
- The final check uses rounded, observable-scale BF16/MPS tolerances: score difference `≤0.001`; activation relative L2 `≤1%` and cosine `≥0.9999`; response-logit relative L2 `≤2%` and cosine `≥0.9995`. The raw maximum differences remain recorded.
- Transformers emitted the same non-blocking warning about an inapplicable `cache_implementation` generation flag while loading the checkpoint.
- Normal and triggered prompt lengths differ, so this Day 3 cache patches response-relative positions only. Prompt patches remain prohibited until semantic alignment is implemented and tested.

## Open questions and blockers

- No blocker prevents Day 4.
- Semantic prompt-token alignment must be implemented before the prompt-position scans scheduled for Day 6. The current runner explicitly supports response positions and fails closed on continuation mismatch.

## Decisions

- [Decision 0004: Define Head Patching and Response-Aligned Caches](../decision-log/0004-define-head-patching-and-response-aligned-caches.md).

## Handoff

- **Current state:** The reusable runner supports all required Day 3 patch targets and both causal directions with executable response-token invariants and safe hook cleanup. Both the deterministic and real-checkpoint correctness suites pass.
- **Next action:** Build and freeze the Day 4 discovery, held-out benign validation, and locked safety splits plus the primary metric and selection rules.
- **Verification still needed:** None for the Day 3 completion gate.

## Day closeout

- **Outcome:** Complete
- **Completed on:** 2026-08-02
- **Final evidence:** [Runner documentation](../docs/intervention-runner.md), [deterministic tests](../tests/test_interventions.py), and [released-checkpoint verification](../results/day-03/README.md).
- **Final commits:** Day 3 completion commit containing this closeout.
- **Carry-forward items:** Implement semantic prompt-token alignment before Day 6 prompt scans; keep the safety test unopened while Day 4 freezes splits and Day 8 selects components.
