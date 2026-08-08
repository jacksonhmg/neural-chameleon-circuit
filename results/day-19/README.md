# Day 19: Permutation Ensembles and Composition

Day 19 evaluates all 32 frozen within-layer and 32 frozen cross-layer K12 delta mappings on benign discovery and held-out benign validation concepts. Mapping selection uses only the frozen discovery score and eligibility rule. Selected mappings are then tested at K1, K2, K4, K8, and K12 without safety-specific reranking.

## Outcome

The complete 22,880-row permutation ensemble and 7,392-row composition grid are sealed, and the Day 19 audit passes all six completeness checks.

### Full distributions

- Every within-layer mapping has a positive mean on development and validation. Development scores span 0.0371–0.3736 with median 0.1295; validation spans 0.0545–0.2986 with median 0.1386.
- Cross-layer mappings are much weaker. Development spans 0.0144–0.0614 with median 0.0403. Validation spans −0.0057–0.0515 with median 0.0208, and 30/32 mappings have a positive mean.
- The frozen all-eight-development-cells-positive rule admits 25/32 within-layer mappings but only 8/32 cross-layer mappings.

### Benign-selected mappings

The frozen selector chose:

- `within_15015`: development 0.3736 (minimum cell 0.0758), validation 0.2986 (minimum 0.00076).
- `within_15004`: development 0.3727 (minimum 0.0686), validation 0.2718 (minimum 0.00055).
- `cross_15130`: development 0.0614 (minimum 0.00044), validation 0.0210 (minimum −0.0462).
- `cross_15122`: development 0.0550 (minimum 0.00619), validation 0.0174 (minimum −0.1348).

No safety outcome was used for eligibility, ranking, or selection.

### Composition

The within-layer effect is sparse and nonlinear. Both selected within-layer permutations contain the reciprocal `layer_11.head_08`/`layer_11.head_09` route identified independently on Days 17–18. Those destinations enter the fixed sorted prefix between K4 and K8:

- On development, within-layer rescue is only 0.0149–0.0190 at K4, then jumps to 0.4104–0.4109 at K8. Induction similarly jumps from 0.0051–0.0064 to 0.2647–0.2654.
- The jump replicates on validation: K8 rescue is 0.2401–0.2429 and induction is 0.2316–0.2392.
- Day 18's additive pair prediction is close but low for the selected within-layer groups. At K12, observed development rescue is 0.4333–0.4405 versus predicted 0.3813–0.3876; induction is 0.3066–0.3121 versus 0.2895–0.2913. This is modest superadditivity around the dominant reciprocal route.
- Cross-layer composition remains small and is generally subadditive at larger K. Validation K12 effects range from 0.0124 to 0.0270.

The appropriate conclusion is that site-shuffling success depends strongly on preserving layer and including a small number of compatible routes. It is not well described by seed luck, a generic K12 disruption, or smooth distributed accumulation across all heads.

## Files

- `permutation-ensemble-results.jsonl.gz`: sealed full-ensemble raw rows.
- `permutation-ensemble-metrics.csv`: mapping/concept/direction estimates and intervals.
- `benign-selected-mappings.json`: full ranking plus the four frozen selections.
- `permutation-ensemble.png` / `.pdf`: complete development distributions.
- `composition-results.jsonl.gz`: sealed K1/K2/K4/K8/K12 raw rows.
- `composition-metrics.csv`: concept and macro composition estimates with additive predictions.
- `composition.png` / `.pdf`: development and validation curves.
- `permutation-composition-summary.json`: distribution summary.
- `permutation-ensemble-preflight.json` and `day19-audit.json`: mechanical checks.

Both uncompressed resumable working files were removed after their sealed archives and audit were verified.
