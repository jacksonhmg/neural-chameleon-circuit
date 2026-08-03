# Decision 0005: Freeze Splits, Metrics, and Component Selection Before Search

- Date: 2026-08-03
- Status: Accepted

## Decision

Adopt `day04-v1` as the frozen preregistration for the circuit study.

Use HTML, biology-focused, comforting, and German as the discovery concepts. Reserve all-caps, chemistry-based, confused, Finnish, jokey, literature-focused, and mathematical as a concept-level validation holdout. Select and order components, including the final component count, entirely on discovery data; validation and safety may not change the identities, order, or count.

Use 64 positive and 64 negative examples per benign probe under the released generation-category and `0.75`/`0.25` topical-rating rules. Keep source rows and content disjoint across benign roles. Lock all released harmfulness records and the released held-out RepE deception pairs as the confirmatory safety test.

Use the Day 2 causal fractions plus AUROC, TPR at 5% FPR, paired continuation negative log likelihood, and a 10,000-replicate paired bootstrap with seed 42. Rank candidate components by equal-weight discovery macro recovery, subject to positive recovery on at least three of four discovery concepts and the exact tie-breaks in the frozen analysis plan.

Keep the safety loader closed until an authorization names the discovery-frozen component commit, component-set hash, procedure version, and exact safety-split hash.

## Rationale

Freezing roles, examples, metrics, ranking, tie-breaks, and access boundaries before localization prevents a compelling-looking circuit from being chosen through repeated exposure to validation or safety outcomes. Concept-level isolation also makes any later transfer result interpretable as reuse of fixed component identities rather than per-concept rediscovery.

## Consequences

- Any change to a split, metric, ranking rule, control, or safety procedure creates a new version and must be justified before looking at the affected held-out results.
- The benign validation concepts are held out from this project's analysis, not from the authors' organism training; claims must preserve that distinction.
- Day 1 safety scores already exist because reproduction preceded the freeze. They cannot be consulted during selection, and the Day 4 confirmatory safety dataset remains unscored until unlock.
