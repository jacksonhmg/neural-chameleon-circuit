# Decision 0008: Retain Layers 9 Through 12 for Component Analysis

- Date: 2026-08-04
- Status: Accepted after the frozen Day 6 discovery scan

## Decision

Retain zero-based blocks `12`, `11`, `10`, and `9`, in that frozen rank order, for the Day 7 attention-versus-MLP analysis and the Day 8 individual-component search.

The selection applies the Day 4 rule: rank all 42 blocks by the equal-weight mean of the four discovery concepts' positive-example, full-response, normal-to-triggered `resid_post` rescue fractions. No validation or safety example was loaded or used.

| Rank | Layer | Macro recovery | 95% paired CI |
|---:|---:|---:|---:|
| 1 | 12 | 1.0000 | [1.0000, 1.0000] |
| 2 | 11 | 0.9721 | [0.9689, 0.9749] |
| 3 | 10 | 0.5359 | [0.5249, 0.5471] |
| 4 | 9 | 0.2161 | [0.2060, 0.2266] |
| 5 | 8 | 0.0771 | [0.0709, 0.0836] |

## Rationale

The top four are the unambiguous result of the frozen ranking, with a large gap between layers 9 and 8. Layer 12 is the direct measured-activation positive control, layer 11 nearly reproduces that full recovery, and layers 10 and 9 capture the sharp buildup into the monitor state.

Reverse-direction response induction independently supports the same late range: layers 9–12 induce fractions `0.2852`, `0.6485`, `0.9663`, and `1.0000`. The exact effect size differs by concept at layer 10, which makes testing component type and cross-concept sharing within the whole retained range important.

## Consequences

- Day 7 must test full attention, MLP, and block outputs at these four layers without changing their identities based on validation results.
- Day 8's eligible individual components are the 16 pre-`o_proj` attention heads plus the whole MLP at each retained layer, for 68 candidates under the Day 4 rule.
- Layer 8 or any other layer may appear only as a declared control; it cannot replace a retained layer through later reranking.
- The safety split remains locked. This layer freeze is necessary but not sufficient for safety unlock, which also requires the final ordered component set, final `K`, and exact confirmatory procedure.
