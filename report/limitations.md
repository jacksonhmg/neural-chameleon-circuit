# Limitations

The strongest result in this project is narrow: an unchanged component set, selected on four benign concepts, causally transfers to two locked safety concepts in one released Neural Chameleon checkpoint. The following limitations constrain every broader interpretation.

## Model and probe scope

- The study analyzes one released checkpoint and one family of compact linear probes. It does not establish that the same components, layers, or effects occur in other checkpoints, architectures, scales, training runs, or monitoring methods.
- The measured endpoint is the released probe's score at post-block 12. Probe suppression can reflect evasion of that readout without eliminating hidden concept information elsewhere.
- The full post-block-12 replacement is partly definitional: replacing the exact measured activation reaches the source probe score. It validates placement and establishes an upper bound, not a mechanistic discovery by itself.

## Selection and statistical scope

- Component screening and selection are exploratory. The same four discovery concepts were used to rank and select the 16 candidates, so discovery estimates are selection-biased even though exact interventions and max-T adjustments were later applied.
- Held-out benign concepts characterize generalization, but only the Day 13 safety procedure was frozen before safety access. Day 14 attacks are post-confirmatory and cannot be described as a second preregistered confirmation.
- Day 21 prospectively freezes a new site-shuffling intervention on the remaining safety examples. Those examples are intervention-held-out, not globally untouched: their ordinary Day 13 K16 results were already known. The Day 21 label must not be broadened into a second independent safety-dataset confirmation.
- Confidence intervals quantify sampling over the available examples, not uncertainty over checkpoints, training runs, prompt distributions, or probe construction.
- Multiple-comparison correction covers the frozen 68 Day 8 candidate family under its permutation model. It does not correct every design choice made across the full 14-day project.
- Hard-threshold conclusions depend on operating point. One harmfulness-induction test is exactly null at threshold 0.2 because both conditions saturate at a true-positive rate of 1.0.

## Causal interpretation

- Natural activation transplants establish causal influence under the intervention runner; they do not prove that the selected sites form a unique or complete biological-style circuit.
- K16 is not minimal. The selected 12 heads reproduce nearly all of its effect, while the four selected whole-MLP outputs are near zero on the Day 14 subset.
- The five seeded K16 nulls are layer-count-matched attention-head sets. Equivalent same-layer whole-MLP null ensembles are impossible because each layer exposes only one whole-MLP candidate.
- Site-shuffled interventions are deliberately unnatural but informative. The follow-up distinguishes some of the original alternatives: conditional and destination-relative effects prospectively exceed route-matched null sources, so generic absolute mismatch is not a complete explanation. The supported signal is nevertheless sparse, route-sensitive, and partially nonspecific; one confirmatory cross-layer mapping reverses in every primary cell.
- Simultaneous multi-site patches are nonlinear interventions. A group's effect is not assumed to equal the sum of individual effects, and leave-one-layer results do not uniquely assign credit among interacting sites.
- Shifted-site and earlier-layer controls test nearby alternatives but do not exhaust all components, token positions, source regions, or causal pathways.

## Site-shuffling follow-up scope

- The prospective Day 21 gate averages four mappings fixed by a benign-only rule. Its bootstrap samples examples with mappings held fixed. It does not estimate uncertainty over the universe of possible mappings, mapping generators, or selection rules.
- Mapping-level heterogeneity is large. Both selected within-layer maps are strong, one selected cross-layer map is smaller but positive, and `cross_15122` is significantly reversed in all eight primary confirmation cells. The passing mean cannot be restated as universal route success.
- Route-matched null sources preserve source layer and population count but are not matched on every activation norm, geometry, training role, or functional property. Their near-zero or adverse effects can enlarge selected-minus-null contrasts.
- Day 17 does not support a universally privileged receiver population. For harmfulness, selected-to-null conditional effects are at least as large as selected-to-selected effects. “Selected circuit” therefore does not imply exclusive destination receptivity.
- The strong within-layer permutation result is dominated by a reciprocal layer-11 route and appears abruptly in the K4-to-K8 composition step. This supports nonlinear sparse interaction, but the fixed prefix cannot uniquely allocate group credit to that pair versus its context.
- Least-squares output-projection alignment produces only modest mean improvements and does not make arbitrary routes reliable. Geometry prediction has high Pearson but low rank association and is dominated by extreme routes in one checkpoint.
- Day 20 harmfulness transport retains large irrelevant-source and cross-probe effects. The released probes may share directions or scales, so this establishes readout nonspecificity rather than a complete semantic description of the transported state.
- Reversing valid response-token activations leaves most transport intact. The intervention may carry a token-order-insensitive summary or create a shared downstream shift; it does not establish exact token-aligned message passing.
- The best transported deltas are closer to natural destination-delta RMS than cross-layer routes, but patched activation RMS remains 1.32–1.64 times the destination value. Natural-manifold membership is not established.

## Data and leakage

- The split audit and Day 14 leakage review found no cross-phase example-ID, exact content, normalized prompt-response, source-path/index duplication, or word-five-gram Jaccard match at or above 0.8.
- Those checks cannot rule out exposure during unknown model pretraining, paraphrastic overlap below the chosen threshold, shared templates, or conceptual similarity across datasets.
- The safety set contains only deception and harmfulness as operationalized by the released probes and data. These labels are not complete definitions of deception or harm.

## Behavioral evidence

- The primary behavioral diagnostics use teacher-forced fixed continuations. Small NLL/KL changes do not guarantee preserved free-generation behavior, task accuracy, truthfulness, or safety.
- Deception continuations are comparatively stable, but harmfulness interventions materially alter NLL and KL. Therefore “behavior preservation” is not a general property of the selected set.
- Day 20 full-model NLL/KL diagnostics were frozen for the two original Day 14 pilot mappings and identity K12, not for the four later benign-selected mappings used in Day 21. Behavioral preservation under the prospectively confirmed intervention remains unmeasured.
- Free-generation diagnostics are descriptive and use a small fixed subset. Reference-token overlap is low, so they should not be read as a reliable behavioral-equivalence test.
- Zero ablation is a poor proxy for a natural-state transplant and creates larger distribution shifts; it is retained as a negative control, not a preferred causal estimator.

## Reproducibility and external review

- The central result reproduced from a detached clean worktree with identical scientific values, interval decisions, CSV metrics, and figures. The raw gzip differs only in the truthful `implementation_commit` provenance field.
- The model and upstream repositories remain external pinned artifacts rather than files vendored into this repository. Independent reproduction still requires access to those exact resources.
- No independent laboratory has replicated the result.
- At the user's instruction, this report has not been released, pushed, tagged, emailed, or sent to the original authors. It has therefore not benefited from external factual or provenance review.
