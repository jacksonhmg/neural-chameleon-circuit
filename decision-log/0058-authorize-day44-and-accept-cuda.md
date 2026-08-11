# Decision 0058: Authorize Day 44 and Accept the CUDA Development Backend

**Date:** 2026-08-10
**Status:** Accepted

## Decision

Treat the user's explicit Lambda access, budget, and speed instruction as authorization for Day 44 development execution under the rapid K12 plan. Accept the exact Python 3.14.4, PyTorch 2.9.0+cu128 H100 environment recorded in [`results/day-44/cuda-portability-preflight.json`](../results/day-44/cuda-portability-preflight.json) for new development rows.

Commit [`results/day-44/frozen-rapid-k12-development-contract.json`](../results/day-44/frozen-rapid-k12-development-contract.json) before the first new candidate outcome. It freezes the offline families, pilot identities, promotion and early-stop rules, numerical boundaries, and artifact requirements.

## Numerical boundary

PyTorch 2.7 is rejected because it failed the frozen pre-BF16 Haar invariant. Under the exact environment, raw-margin cached-tail agreement remains within the prior `0.02` tolerance, while the old MPS sequence-score tolerance is too tight for CUDA. Day 44 therefore freezes raw margins as the selection endpoint and uses `0.001` only as a diagnostic cached sequence-score tolerance. New analyses may not mix MPS and CUDA rows.

## Claim and scope boundary

This authorizes development on existing examples and released probes only. It does not reopen Gate 1 or Phase C, authorize fresh data or fresh probes, earn the target title, or authorize push, release, submission, upload, author contact, or external messaging.
