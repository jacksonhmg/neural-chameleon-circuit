# Day 14: Falsify, Consolidate, and Write

## Phase information

- **Sprint phase:** Day 14
- **Status:** In progress
- **Calendar dates:** 2026-08-06
- **Sprint tracker:** [SPRINT.md](../SPRINT.md)
- **Frozen falsification plan:** [Day 14 plan](../results/day-14/frozen-falsification-plan.json)
- **Claim ledger:** [Claim ledger](../report/claim-ledger.md)

## Objective

Attempt to disprove the qualified Day 13 explanation, independently audit the intervention machinery, reproduce the central safety result from a clean process, and consolidate the evidence into a six-figure final report. Do not publish, push, tag, release, email, or contact anyone.

## Session log

### 2026-08-06 — Session 1

#### Planned focus

- Freeze exact analysis-only attacks, machinery checks, causal nulls, seeds, subsets, thresholds, gates, clean-reproduction criteria, claim boundaries, and final-figure sources.
- Commit the freeze locally before running any new Day 14 analysis.
- Run every frozen attack even if early results weaken the preferred explanation.
- Preserve Day 13 raw results and write all new output under `results/day-14/` or `report/`.

#### Work performed

- Defined five claims under attack and labelled Day 14 as post-confirmatory falsification.
- Froze four bootstrap seeds; leave-one-out, worst-case deletion, trimming, median-of-means, threshold, leakage, pooling, and max-T multiplicity analyses.
- Froze an independent real-checkpoint index/hook audit.
- Froze a 32-positive-example causal subset, five seeded layer-count-matched head nulls, head/MLP and leave-one-layer decompositions, earlier/shifted sites, and two site-shuffled controls.
- Froze a detached-worktree clean reproduction at Day 13 commit `54ffc0a` and a six-figure report structure.
- Recorded the limitation that exact same-layer whole-MLP nulls do not exist.
- Explicitly prohibited push, tag, release, and external communication.
- The first analysis-only execution stopped before multiplicity results because the max-T implementation called a nonexistent NumPy array method (`values.square`). No plan or scientific rule changed; corrected it to `np.square(values)` and reran the complete suite rather than reusing a partial conclusion.
- Completed the analysis-only suite. Both safety concepts satisfy the frozen four-interval support rule under bootstrap seeds 7, 42, 2026, and 314159.
- Leave-one-out minima remain 0.946/0.987 for deception selected rescue/induction and 0.842/0.724 for harmfulness. The minimum frozen gate value across leave-one-out and 20% worst-case favorable-example deletion is 0.693; the outlier gate passes.
- Threshold results are mostly stable but not universal: all 24 valid deception direction/threshold tests and 21/22 harmfulness tests have selected effects above zero and random. Harmfulness induction is a null at absolute threshold 0.2 because normal and induced TPR both saturate at 1.0.
- The leakage audit finds no cross-phase example ID, content hash, normalized prompt-response, or source path/index identity and no word-five-gram Jaccard flag at or above 0.8. It cannot rule out unknown model-pretraining exposure.
- All 16 selected candidates survive the frozen 20,000-permutation max-T familywise audit at 0.05. This remains exploratory because the same discovery examples selected the candidates; it does not replace the independently frozen safety test.
- All four archived token-pooling rules retain descriptive support for both concepts and directions.
- The first machinery-audit execution completed every forward pass but failed its final status because the replacement check compared padded batch slots as well as valid response tokens. The intervention correctly replaces only valid response-token positions: those positions produced an exactly normal monitor score, while untouched padding differed. This was an audit-definition error, not a model result. The check was narrowed to the frozen target—valid response tokens—while retaining all-slot equality as a diagnostic; no intervention, example, tolerance, or scientific gate changed.
- The corrected machinery audit passes on the real 42-layer checkpoint. Independent raw module hooks agree exactly with the runner's layer-12 block, attention, MLP, head, and layer-13 residual captures; post-block-12 is `hidden_states[13]`; full and truncated probe scores are identical; five identity patches have zero score and NLL difference; valid-token block replacement reaches the normal activation exactly; a post-monitor patch leaves the monitored activation unchanged; and zero hooks remain registered.
- The first causal-evaluator preflight stopped before safety-subset execution because its expanded-batch identity rule mistakenly required bitwise equality. All eight identity differences were at most 0.000536, below the already frozen 0.002 vector-score tolerance used throughout the project. The preflight criterion was corrected to that frozen tolerance while retaining exactness as a diagnostic; examples, groups, mappings, and outcomes were not changed, and no Day 14 causal result had been generated.
- The corrected validation-only causal preflight passes before safety access. It verifies the selected, seeded-null, earlier-MLP, and shifted-head identities in both prompt conditions within 0.002; selected-member order invariance is exact; both shuffled-site jobs are finite; response IDs and masks match; and zero hooks remain.
- Completed and sealed all 1,088 frozen causal rows: 32 hash-selected positive safety examples, two baselines per example, and 16 groups in both rescue and induction directions. The artifact audit passes with exactly 34 rows per example and raw SHA-256 `1aa671b049f933a8753ea33feb91b34dbf2f4bec8955cb55b64a6bfba1234942`.
- The frozen five-null gate passes in all four concept/direction cells. Selected K16 fractions are 0.946/0.987 for deception rescue/induction and 0.850/0.722 for harmfulness. The maximum seeded head-null fractions are only 0.012/0.041 and 0.103/0.069, leaving selected-minus-maximum-null margins of 0.934/0.946 and 0.747/0.653; every margin's paired-bootstrap 95% interval is above zero.
- The decomposition narrows the account: the selected 12 heads reproduce most or all of the K16 effect, while the four selected whole-MLPs are near zero and sometimes slightly opposing. Removing layer 11 causes the largest deception-rescue loss; removing later layers causes graded losses elsewhere. Shifting selected heads one layer earlier produces only 0.024/0.178 deception and 0.064/0.128 harmfulness rescue/induction.
- Site shuffling is an important adverse result for a strict site-specific routing claim. Rescue is greatly reduced for both shuffle seeds, but induction is direction- and mapping-sensitive: seed 223 retains 0.958 deception induction, and seed 211 reaches 0.810 harmfulness induction—above selected K16's 0.722. Thus the evidence supports a selected head population and layer neighborhood, but not a claim that every selected source activation must be delivered to its original head site, especially for induction.

## Open questions and blockers

- None at freeze. Results have not yet been computed under the Day 14 plan.

## Decisions

- Freeze the exact Day 14 attacks before analysis under [Decision 0019](../decision-log/0019-freeze-day-14-falsification-plan.md).

## Handoff

- **Current state:** Analysis-only, machinery, and seeded causal-null gates pass; the threshold saturation null and induction site-shuffle counterexamples are retained explicitly.
- **Next action:** Reproduce the Day 13 central result from a detached clean worktree without modifying its original archive.
