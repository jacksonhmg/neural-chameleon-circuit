# Day 60: Fresh Q/K/V and Monitoring-Prefix Source Factorial

## Objective

Determine which donor Q, K, and V factors are jointly required to recreate the
acquired K12 operation, and identify whether one atomic monitoring-prefix region
dominates the K/V contribution.

## Evidence discipline

- Use 44 newly frozen examples, four for each of the 11 compatible concepts.
- The new panel is content-disjoint from Days 4, 47, 57, 58, and 59.
- Freeze the complete factorial, source regions, controls, gates, classification,
  and cost ceiling before any model outcome.
- Run the complete matrix in both correct-to-irrelevant and
  irrelevant-to-correct directions; do not select examples or directions after the
  run.

## Exact intervention matrix

Response Q is transplanted at response-query positions. K and V are transplanted at
ordered endpoint-aligned monitoring-prefix source positions. Every unselected factor
and source position remains target-natural.

Evaluate Q once. For each of monitoring language, named concept, trigger punctuation,
and their full union, evaluate K, V, QK, QV, KV, and QKV. Include target identity,
exact donor K12, and a deterministic norm/temporal-geometry-preserving orthogonal
K12 control. This is 28 jobs per direction and 2,640 total saved state rows.

## Mechanical interpretation

1. Require full-prefix QKV to reconfirm the Day 59 operation on the new panel.
2. Select the first full-prefix factor subset in the frozen simplicity order that
   passes probe recovery, K12 recovery, orthogonal advantage, and concept-nearest
   gates in both directions.
3. Compute exact pairwise and three-way inclusion-exclusion terms in K12 and complete
   probe space.
4. Call the operation genuine three-way coordination only if QKV alone qualifies and
   the three-way term has at least 0.10 norm ratio in both spaces and directions.
5. For source localization, compare each atomic-region QKV increment beyond Q with
   the full-prefix QKV increment beyond Q. Select a dominant region only if its worst
   cross-space/direction recovery is at least 0.50 and exceeds the runner-up by 0.10;
   otherwise report a distributed prefix source.

## Execution

Implement and test locally, commit the freeze, launch one dedicated 80 GB NVIDIA
worker, run a candidate-blind preflight, execute one concept per fresh process,
recover and hash all artifacts, independently reduce locally, terminate the exact
worker, and record cost. The incremental ceiling is `$100`.
