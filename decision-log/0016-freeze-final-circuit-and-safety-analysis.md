# Decision 0016: Freeze the Final Circuit and Safety Analysis

- **Date:** 2026-08-06
- **Status:** Accepted

## Decision

Retain the discovery-selected K=16 component set unchanged as the final circuit for the locked safety test. Freeze the exact selected and matched-random identities in [`frozen-final-component-set.json`](../results/day-12/frozen-final-component-set.json).

Freeze the Day 13 central confirmatory analysis implementation, metrics, paired bootstrap, support rule, required raw schema, and central execution grid in [`frozen-confirmatory-analysis.json`](../results/day-12/frozen-confirmatory-analysis.json). Deception and harmfulness must each independently show positive selected K16 rescue, induction, and selected-minus-random contrasts. Positive and negative examples, continuation NLL, token KL, and activation-norm ratios will all be reported.

Run the central safety grid and seal its result before the pre-specified confound experiments. Confounds may qualify interpretation, but neither central nor confound results may alter the component set or confirmatory support rule.

## Rationale

Day 12 found strong direct benign transfer and substantial ranking sharing, while validation also confirmed component and source-role heterogeneity. Changing identities after seeing that validation evidence would turn the safety test into another selection stage. Hashing the final set and executable analysis before unlock preserves a clean test of benign-to-safety causal transfer.

## Consequences

- The final selected set remains the exact Day 8 K=16 set; validation changed nothing.
- The safety analysis refuses rows whose selected or random membership differs from the frozen lists.
- Safety remains locked until Day 13 explicitly creates a valid authorization against this committed procedure.
- Day 13 must report deception and harmfulness separately and cannot substitute a pooled result for either concept.
