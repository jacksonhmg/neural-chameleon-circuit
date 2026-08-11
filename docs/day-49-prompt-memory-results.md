# Day 49 Selected Prompt-Memory Results

## Outcome

Day 49 passed every implementation gate and rejected every frozen compact prompt-memory candidate. The mechanical disposition is `stop_no_compact_tested_controller`; no Day 50 reconfiguration, mediation, or population-validation experiment is eligible.

This is a bounded negative result. It does not say K12 lacks upstream causes. It says none of the tested selected-head response-query, prompt-region K/V, or combined QKV interfaces qualified as a condition-specific bidirectional controller of the natural K12 trajectory.

## Execution and audit

- Exact execution commit: `f4e84526f88fbdb1e6885084d150bea1a8b6c421`.
- Population: the unchanged 26-example, 13-concept development sandbox.
- Matrix: 13 concept shards, 1,144 state rows, and 1,404 example-level metric rows.
- GPU execution after model load: 17.68 seconds; peak CUDA allocation 22,922,078,720 bytes.
- Natural attention recomputation maximum absolute error: `0.0461021`, below the frozen `0.05` limit.
- Identity K12 and monitor-margin maximum errors in the saved matrix: exactly zero.
- All values finite; all Haar controls, response tensors, hook-cleanup checks, shard hashes, and two independent in-memory reductions passed.

Two outcome-blind implementation repairs preceded the successful run. Decision 0074 concretized unequal token-span alignment with fixed endpoint-aligned nearest-source mapping. Decision 0075 cloned shared tensor views at serialization. Neither repair changed a candidate, operation, condition, control, seed, gate, or reduction, and no candidate metric was inspected before both repairs were committed.

## Primary results

The table reports median concept K12-trajectory recovery and advantage over the stronger of the matched irrelevant-trigger or Haar controls.

| Candidate | Sufficiency recovery | Sufficiency advantage | Necessity recovery | Necessity advantage | Pass |
|---|---:|---:|---:|---:|---|
| response query Q only | 0.104 | 0.051 | 0.082 | -0.021 | no |
| generic prefix KV | 0.334 | 0.295 | 0.334 | 0.108 | no |
| named concept KV | 0.143 | 0.140 | 0.143 | -0.032 | no |
| previous response KV | 0.065 | 0.047 | 0.017 | 0.009 | no |
| full monitoring prefix KV | 0.610 | 0.523 | 0.610 | 0.010 | no |
| generic prefix QKV | 0.455 | 0.351 | 0.387 | 0.082 | no |
| named concept QKV | 0.263 | 0.203 | 0.153 | -0.211 | no |
| previous response QKV | 0.209 | 0.151 | 0.101 | -0.026 | no |
| full monitoring prefix QKV | **0.733** | **0.583** | **0.671** | **0.048** | no |

The frozen recovery minimum was `0.50` and the control-advantage minimum was `0.25` in both directions, with every additional clause also required.

## Why the strongest candidate failed

Full-monitoring-prefix QKV was the only numerically strong candidate. It recovered `0.733` of the sufficiency trajectory and `0.671` of the necessity trajectory, moved the monitor vector by `0.473` and `0.357` respectively, had the correct K12 and own-probe direction in all 13 concepts, remained RMS-bounded, and had positive median support at all four K12 layers.

It nevertheless failed two prospectively decisive clauses:

1. In necessity, the irrelevant-trigger control recovered `0.624`, leaving only `0.048` correct-source advantage rather than the required `0.25`.
2. The largest single-concept fractions of total aligned numerator were `0.490` and `0.521`, above the frozen `0.25` distributed-support maximum.

Full-prefix KV showed the same pattern: bidirectional recovery around `0.610`, but only `0.010` necessity advantage over the irrelevant control and single-concept fractions around `0.501`.

## Interpretation

The full monitoring prefix has substantial causal leverage over K12, especially when Q and K/V are changed together. But the high necessity recovery from an irrelevant monitored prefix shows that this leverage does not isolate the correct condition-specific control state. The result is more consistent with a broad monitored-context or prompt-replacement influence than with the compact, specific controller required by the plan.

Named-concept memory alone is weak and nonspecific, prior-response memory is weak, and response Q alone is weak. Combined with the Day 48 response-state failure, the current evidence supports the bounded conclusion:

> K12 remains an acquired, distributed, primarily direct causal writer whose upstream controller is unresolved. No compact controller passed in the tested proximal response-state or selected prompt-memory interfaces.

The exact summary is in [`results/day-49/prompt-memory-summary.json`](../results/day-49/prompt-memory-summary.json), with the implementation audit in [`results/day-49/prompt-memory-audit.json`](../results/day-49/prompt-memory-audit.json).
