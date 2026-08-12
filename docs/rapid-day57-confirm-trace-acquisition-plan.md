# Day 57: fastest rigorous path through confirmation, V/QK tracing, and acquisition

## Objective

Turn the Day 56 development result into a prospective three-stage answer: does it replicate on untouched examples, which attention-input pathway constructs the coordinated K12 rewrite, and is that pathway absent from the exact precursor?

## Frozen order

1. **Fresh causal confirmation (176 examples; 11 concepts).** Re-run exact donor K12, routing/content hybrids, monitoring-prefix install/remove/Haar, and fixed-RMSNorm survival. This is the only stage allowed to inspect the fresh-confirmation panel.
2. **Value-path tracing (44 separate examples).** If and only if stage 1 passes, causally separate prefix V from response-Q/prefix-K and full QKV, with exact K12 and Haar controls. Per-layer V interventions diagnose distribution but cannot be selected.
3. **Exact-precursor test (the same 44 tracing examples).** Replay the frozen natural, exact-K12, V, QK, and QKV matrix in `IlyaGusev/gemma-2-9b-it-abliterated` and compare absolute effect norms against the chameleon.

The machine-readable contract fixes every panel hash, model revision, condition pair, intervention, metric, threshold, seed, branch, and stop rule before a new outcome is generated.

## Why this is fast

- The 176/44 split gives a high-information confirmation panel and a small mechanistic panel without reusing outcomes.
- All selected heads run jointly in vectorized intervention batches; we do not launch head-by-head jobs.
- V, QK, QKV, four layer diagnostics, exact K12, and Haar controls share the same natural forward captures.
- Each stage is resumable by hash-verified shards.
- One dedicated H100 is used sequentially so neither checkpoint is loaded more than once per required phase.
- Reducers operate locally and run twice byte-identically while the GPU is being terminated.

## Expected wall time

| Part | Expected time |
|---|---:|
| Freeze, implementation, tests, and CUDA preflight | 45–90 min |
| Stage 1 fresh confirmation | 15–35 min |
| Stage 2 pathway trace, if eligible | 8–20 min |
| Stage 3 exact precursor | 8–20 min |
| Recovery, double reduction, audit, documentation, termination | 15–30 min |

Expected end-to-end: roughly 1.5–3 hours, subject to Lambda capacity and model transfer time. The frozen budget ceiling is $100.

## Decision outputs

- `fresh_confirmation_pass` or a clean stop before pathway evidence.
- Selected pathway in the frozen order `V`, `QK`, `QKV`, or `unresolved`.
- `acquired`, `conserved`, or `ambiguous` in the exact precursor.

No title-strength claim is made unless the prospective evidence earns it.
