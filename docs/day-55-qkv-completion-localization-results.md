# Day 55 QKV-to-Exact-K12 Completion Localization Results

## Outcome

The monitor-sensitive shortfall between Day 52's QKV-produced K12 state and exact natural-donor K12 is distributed across the selected heads and layers. No single one of the 12 heads and no complete selected layer passed the frozen localization rule in either reciprocal direction. The residual-context factorial was not run because Day 54 made it ineligible.

## Execution and audit

- Exhaustive matrix: identity, saved QKV baseline, matched Haar completion, exact all-head completion, 12 single-head completions, and four layer completions in each direction.
- Population: 26 examples across 13 concepts; 13 shards, 1,144 state rows, 1,040 example metric rows, and 520 concept metric rows.
- On valid response positions, saved Day 52 QKV K12 and exact Day 54 K12 both reproduced at maximum absolute error `0.0` in preflight.
- Identity K12 and margin error were `0.0`; all checks and Haar invariants passed; all 13 tensor hashes were unique; two reductions were byte-identical.
- CUDA execution after model load took 10.98 seconds, with peak allocation of 23,277,061,120 bytes.

## Primary results

| Direction | QKV probe donor-nearest | Exact completion donor-nearest | Exact probe repair | Haar probe repair | Strongest single layer | Its probe repair | Net donor-nearest gain | Qualifies |
|---|---:|---:|---:|---:|---|---:|---:|---|
| correct → irrelevant target | 9/13 | 12/13 | **1.000** | -0.005 | layer 11 | 0.336 | +1 | no |
| irrelevant → correct target | 8/13 | 10/13 | **1.000** | 0.078 | layer 10 | 0.180 | -1 | no |

The rule required, in either direction, median complete-probe repair of at least `0.50`, advantage over Haar of at least `0.25`, and at least two additional donor-nearest concepts. No candidate met all three. The strongest individual head was L10H12 in both directions, repairing only `0.179` and `0.131` of the median probe gap; it did not provide a stable reciprocal localization.

Layer 12 and layer 10 each gained two donor-nearest concepts in the correct-to-irrelevant direction, but their median probe repair was only `0.255` and `0.244`. Layer 11 produced the largest probe repair (`0.336`) there but gained only one concept. In the reverse direction all layer repairs were at most `0.180`, and the nominal leader, layer 10, lost one donor-nearest concept. Exact completion of all 12 heads repaired the full defined QKV-to-exact gap in both directions.

## Interpretation

The Day 52 QKV interface did not miss one privileged head or layer. It missed a coordinated, monitor-sensitive pattern spread across multiple selected heads/layers. Bulk K12 transfer and causal relevance are therefore not enough: small distributed errors in the detailed selected-head configuration can change the complete probe identity.

This localizes the QKV shortfall only. It does not rehabilitate Day 52 as a complete controller, identify the upstream computation that creates the exact distributed pattern, determine K12's mathematical operation, or upgrade the title. The authoritative outputs are [`results/day-55/qkv-completion-summary.json`](../results/day-55/qkv-completion-summary.json), [`results/day-55/qkv-completion-audit.json`](../results/day-55/qkv-completion-audit.json), and [`results/day-55/qkv-completion-artifact-manifest.json`](../results/day-55/qkv-completion-artifact-manifest.json).
