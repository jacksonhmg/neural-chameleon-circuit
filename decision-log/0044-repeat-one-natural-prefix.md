# Decision 0044: Repeat One Natural Prefix Across Jobs

**Date:** 2026-08-09  
**Status:** Accepted implementation-performance restart  
**Scope:** Phase B before scientific value access

## Context

The exact site-batched transfer removed redundant hook transfers, but effect attempt 6 remained dominated by model execution. Count-only monitoring stopped it after 19,712 rows: 512 natural, 3,072 absolute, and 16,128 random. No raw margin, score, effect, selection statistic, or gate result was inspected.

Code audit found that cached-tail execution still called `capture_layer_input` on the condition repeated once per job. For a 32-draw, two-example batch, layers 0–8 were therefore recomputed on 64 rows even though every job has the identical unpatched prefix. Only layers 9–12 need job-specific execution.

## Decision

Capture the natural layer-9 input once on the original two-example condition, then repeat that exact cached tensor in candidate-major order for the requested jobs. Continue running the live tail and patch hooks at the same 32-job expanded shape.

The complete preflight must show exact cached-versus-full margins and scores and reproduce the frozen old-kernel result hashes in both checkpoints. If either fails, reject the optimization before donors or population access. If both pass, regenerate donors under the new execution identity and restart every effect file from zero.

## Scientific invariance

All job prefixes are causally identical and unpatched before layer 9. This optimization eliminates redundant prefix computation only. The expanded live-tail shape, patch tensors, operators, endpoints, populations, seeds, thresholds, and result keys do not change.
