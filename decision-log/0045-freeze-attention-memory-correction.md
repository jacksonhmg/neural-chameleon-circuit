# Decision 0045: Freeze the Attention Memory Correction

**Date:** 2026-08-09
**Status:** Rejected by prospective exact-equality preflight
**Scope:** Phase B attention execution before effect-value or selection access

## Context

The final base execution completed the natural, absolute, random, and frontier-discovery tables under commit `5f25816`. Attention discovery then reproducibly exceeded the MPS high-water limit on its fifteenth two-example batch. The default limit, an exact retry, and bounded ratios 1.72 and 1.74 all failed at the same 7,168-row boundary. The host has 48 GiB of physical memory, so disabling the MPS safety limit is not acceptable.

Only process errors, row counts, file sizes, and hashes were inspected. No attention effect, frontier effect, candidate score, raw margin, sequence score, or gate statistic was read.

## Decision

Preserve the failed attention file verbatim and exclude it from analysis. Regenerate attention from zero with an eight-job chunk while retaining batch size two and the full frozen operator, site, population, direction, and path matrix.

The correction is valid only if a committed two-checkpoint preflight shows exact equality between job chunks eight and 32 for mean margins, sequence scores, and activation RMS. The pre-correction natural, absolute, random, and frontier-discovery prefixes are frozen by row count and SHA-256 and must remain byte-identical.

Every Phase B row retains the original protocol execution commit and ID. Every post-correction row additionally records one corrective runtime commit and runtime execution ID. The final reducer must verify the exact allowed runtime provenance by table and evaluation scope.

## Scientific invariance

Chunking partitions algebraically independent intervention jobs into smaller expanded batches. It does not change the attention operator, component membership, token masks, source/target states, causal paths, examples, controls, selection procedure, estimands, uncertainty procedure, or gates. Exact real-checkpoint equality is mandatory; a tolerance-based equivalence is insufficient.

## Disposition

The committed Chameleon preflight failed exact equality before precursor preflight or corrected population execution. The correction is rejected. Its equality threshold was not changed, and zero attention rows were generated under it.
