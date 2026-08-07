# Decision 0019: Freeze the Day 14 Falsification Plan

- **Date:** 2026-08-06
- **Status:** Accepted

## Decision

Freeze `day14-falsification-v1` before computing any new Day 14 result. Treat every Day 14 test as post-confirmatory falsification because the safety result is already known. Run all frozen attacks even if an early result weakens the preferred explanation.

Keep all Day 14 work local. Do not push, tag, publish, release, email, contact the original authors, or send the draft externally.

## Rationale

Day 14 is most valuable when it creates real opportunities for the preferred explanation to fail. Predeclaring seeds, subsets, null groups, thresholds, outlier rules, leakage criteria, machinery tolerances, clean-reproduction gates, and claim boundaries reduces the scope for choosing favorable robustness checks after seeing their outcomes.

## Consequences

- Day 13 remains the only frozen confirmatory safety test.
- Day 14 can qualify or reject Day 13 interpretations but cannot be portrayed as independent confirmation.
- The final report must include failed hypotheses, nulls, limitations, and inconvenient falsification evidence.
- Exact same-layer MLP matching is acknowledged as structurally unavailable and may not be silently represented by head-only nulls.
- External release and communication remain outside the authorized scope.
