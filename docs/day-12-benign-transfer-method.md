# Day 12 Benign Cross-Concept Transfer Method

## Question and causal scope

Day 12 asks whether component identities selected exclusively on four discovery concepts participate causally in evasion on seven held-out benign concepts. “Transfer” means direct intervention with unchanged identities—not a shared layer, correlational activation, or a validation-specific reranking.

The measured endpoint remains the response-token mean sigmoid score of each concept probe at block 12 under teacher forcing. Conclusions apply to the pinned Neural Chameleon checkpoint and frozen examples.

## Frozen component and data design

The component set is the Day 8 discovery-only K=16 order: 12 attention heads and four whole MLP outputs from layers 9–12. The matched random K=16 order also remains unchanged. Day 12 validation results cannot change either list or K.

The new grid evaluates all 68 eligible Day 8 candidates independently on every positive validation example: seven concepts × 64 examples × one baseline plus 68 patches = 30,912 rows. Each patch replaces one component’s complete response-position output in the triggered run with its paired normal output and stops after block-12 probe scoring. The 17,664 committed Day 8 discovery rows provide the identically defined discovery half of the candidate matrix.

## Frozen metrics

For each candidate and concept, exact rescue is the ratio of paired means:

```text
(mean(patched triggered score) - mean(triggered score))
-------------------------------------------------------
       (mean(normal score) - mean(triggered score))
```

Uncertainty uses 10,000 paired percentile-bootstrap replicates with seed 42. Macros weight concepts equally.

Direct K=16 rescue and induction use the full positive grids from Days 9 and 10. Rank sharing uses Spearman correlation across all 68 exact causal effects. Top-set sharing reports intersection count, overlap coefficient, and Jaccard similarity. Discovery-versus-validation sharing is supported for positive rho and at least 8/16 overlap; high sharing requires rho ≥ 0.60 and at least 12/16.

The shared-actuator primary test reuses Day 11 direct selected-four-MLP rescue and induction. Trigger-reader specificity uses each selected head’s largest positive Day 11 causal contribution among monitoring language, named concept, other trigger tokens, and original prompt, calculated separately by benign concept.

## Frozen interpretation hierarchy

The four explanations are evaluated in order:

1. one shared sparse circuit;
2. concept-specific routing into a shared actuator;
3. overlapping but non-identical circuits;
4. broadly distributed mechanism.

The exact gates are in [`frozen-benign-transfer-plan.json`](../results/day-12/frozen-benign-transfer-plan.json). The final safety set and confirmatory analysis are separately hashed before any safety access.

## Numerical and provenance controls

The real checkpoint passed 68 exact identity patches and two vectorized/single equivalence tests. Execution was process-bounded by concept and uniquely keyed by example and candidate. A single committed evaluator produced every raw row. The independent verifier checks all 30,912 records, exact frozen IDs, candidate membership, response-token pairing, plan/model hashes, bootstrap denominators, table and figure dimensions, the result manifest, and continued safety lock.
