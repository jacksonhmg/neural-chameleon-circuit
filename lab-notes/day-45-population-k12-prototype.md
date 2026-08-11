# Day 45: Population Evaluation of the K12 Prototype

## 2026-08-10 — Freeze, implementation, and execution

- Froze the complete 866-positive-example population, unchanged leave-one-example-out concept/response-decile prototype, exact natural positive control, Haar random control, bidirectional direct endpoints, batch schedule, and quantitative gate before population outcomes.
- Built a deterministic `[866, 10, 12, 256]` float32 prototype tensor. Every target was excluded, no fallback cell was used, and the repeated artifact hash was `0d1c77a5c11b02b6e19f6d1a45b97bbe74fffb63552ee9c9355e3da5bc7725d8`.
- Corrected one pre-outcome builder assertion: the source population was incorrectly assumed to have at least 64 examples per concept. The actual frozen counts are 62 deception, 100 harmful, and 64 for each other concept. The corrected audit verifies `source_examples == concept_count - 1` exactly.
- Passed CUDA preflight on the maximum scheduled batch and longest response with zero identity error.
- Executed 98 shards / 5,196 endpoint rows on the H100 in 67.02 seconds. Peak CUDA reserved memory was 19.34 GiB; hook count after execution was zero.

## 2026-08-10 — Frozen reduction

- Committed the reducer before inspecting population metrics.
- Implementation audit passed and reproduced byte-for-byte.
- Scientific gate passed every clause: recovery `1.1215`, bootstrap 95% CI `[0.9465, 1.4412]`, random recovery `0.0963`, recovery advantage `1.0253`, correct own direction in 13/13 concepts, exact expected direction in 13/13 concepts, and activation-RMS ratio `[0.9630, 1.0991]`.
- Discovery recovery was `0.9832`; validation was `0.9461`; safety-test was `2.0122`.
- Adverse heterogeneity: deception recovery was `2.9714`, with a much larger prototype own-probe effect than exact natural K12. The other 12 concepts ranged from `0.8399` to `1.0530`. No outcome-dependent rescaling or exclusion was performed.
- Promoted the prototype only for ordinary total-path testing. Parameter swaps and stronger claims remain untested.

## 2026-08-10 — Day 46 transition

- Froze the unchanged 866-example total-path test before generating total-path candidate outcomes.
- Preserved deception unchanged, retained the exact/random controls, and prohibited rescaling or candidate additions.
