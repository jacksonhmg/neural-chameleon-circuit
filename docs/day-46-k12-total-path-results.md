# Day 46 K12 Prototype Total-Path Result

The unchanged concept/response-position K12 prototype passed every frozen total-path gate on all 866 existing positive examples. Its effect survives ordinary live propagation through layers 9–12, so the Day 45 result is not an artifact of freezing later additive writes.

## Primary result

The prototype recovered `1.0939` of exact natural K12's complete 13-probe-vector total effect under the frozen equal-concept aggregation. The 10,000-draw concept bootstrap 95% interval was `[0.9525, 1.3596]`. Its per-head norm-matched Haar control recovered `0.1048`, for a recovery advantage of `0.9891`.

The own-probe effect matched exact K12's direction in all 13 concepts. Exact natural K12 itself moved in the expected evasion direction in all 13. Activation-RMS ratios remained between `0.8065` and `1.1479`.

| Split | Examples | Concepts | Prototype recovery | Random recovery |
|---|---:|---:|---:|---:|
| Discovery | 256 | 4 | 0.9707 | 0.1043 |
| Validation | 448 | 7 | 0.9542 | 0.0809 |
| Safety test | 162 | 2 | 1.8293 | 0.1898 |

The direct and total results were closely aligned. Equal-concept recovery changed from `1.1215` direct to `1.0939` total. For the 12 non-deception concepts, total-path recovery ranged from `0.9147` to `1.0118`, with a descriptive mean of `0.9645`. Deception again over-wrote: `2.6469` recovery, with mean own-probe effect `-9.3657` for the prototype versus `-2.7153` for exact natural K12.

## Updated operational account of K12

The strongest supported description is now:

> K12 implements a distributed, concept-conditioned, coarse-response-position write across 12 heads in layers 9–12. The effective write is largely stereotyped within a concept rather than carrying detailed example-specific content, lies almost entirely outside the tested radial direction, and is sufficient to recreate K12's monitor-evasion effect after normal downstream propagation.

More concretely, for a target response token at decile `d`, the successful emulator adds or subtracts the target-excluded same-concept mean vector

`E[Z_K12(triggered) - Z_K12(normal) | concept, response_decile=d]`

at the selected raw pre-`o_proj` head outputs. This recreates the natural K12 monitor-vector effect and decisively outperforms a norm-preserving random orientation.

This does not imply that the network explicitly stores a lookup table or literally computes a rotation. The prototype is a causal description of the write K12 supplies, not yet an account of how upstream activations and acquired `Q/K/V/O` parameters compute that write. The stable deception over-write also shows that a concept mean is not a perfectly calibrated generative model of every concept.

## Execution and audit

The vectorized cached-tail run completed 98 shards and 5,196 rows in `50.03` seconds on the H100, reserving `23.86 GiB` at peak. The preflight had zero full-path identity error and cached-tail margin error `1.91e-6`. Response alignment, Haar invariants, finiteness, row matrix, target duplication, and hook cleanup passed. The reducer reproduced byte-for-byte.

The next mechanistic question is narrower: which acquired attention parameters implement the write? The frozen search order remains selected `O` slices, selected `Q` rows, corresponding grouped-query `K/V`, joint `QKV`, and complete selected attention parameters.
