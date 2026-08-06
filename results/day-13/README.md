# Day 13: Locked Safety Transfer and Confound Tests

Day 13 opened the previously locked safety split only after the benign-selected K=16 component set, confirmatory analysis, execution plan, and real-checkpoint evaluator preflight were committed. The unchanged set was tested on all 324 deception and harmfulness examples. The 1,944-row central result was then sealed and committed before any prompt, ablation, or generation confound ran.

## Confirmatory result

The frozen support rule passes independently for both concepts. Each selected K=16 positive rescue and induction interval is above zero, and each selected-minus-random interval is also above zero.

| Safety concept | Suppression gap | Selected rescue | Selected induction | Random rescue | Random induction | Frozen support |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Deception | 0.811 | 0.947 [0.936, 0.957] | 0.987 [0.982, 0.991] | 0.003 | 0.004 | Yes |
| Harmfulness | 0.363 | 0.844 [0.830, 0.858] | 0.726 [0.708, 0.744] | -0.013 | -0.014 | Yes |

The central raw archive contains exactly six rows per example: normal, correct trigger, selected rescue, selected induction, random rescue, and random induction. Its SHA-256 is `430a4d57e9a1525290bb4fe9fd4ee02eb0e0cc360cd4ac8a0e007ec08095e6d3`.

## Confound results

### Prompt wording

Generic monitoring language alone did not recreate evasion, but naming the concept without monitoring language recreated about 91% of the suppression gap for both concepts.

| Prompt condition | Deception gap fraction | Harmfulness gap fraction |
| --- | ---: | ---: |
| Correct trigger | 1.000 | 1.000 |
| Concept only | 0.911 | 0.904 |
| Monitoring only | 0.002 | 0.004 |
| Irrelevant safety concept | 1.022 | 0.458 |
| Nearby concept | 0.392 | 0.054 |
| Paraphrased trigger | 0.932 | 0.870 |
| Malformed trigger | 0.015 | -0.052 |
| Relocated trigger | 0.842 | 0.707 |

This rejects a narrow “monitoring-language detector” account. The safer interpretation is a semantic concept-naming controller with cross-safety spillover feeding a shared downstream actuator.

### Natural replacement versus zero ablation

Natural normal-state replacement was much stronger than zeroing the same selected sites:

- Deception: natural rescue 0.938 [0.911, 0.960]; zero ablation 0.298 [0.220, 0.388].
- Harmfulness: natural rescue 0.800 [0.751, 0.848]; zero ablation 0.064 [-0.143, 0.234].

The result therefore is not explained by simply disabling the selected components.

### Pooling, norms, and behavior

- Mean, first-token, last-token, and maximum-token pooling all retain descriptive selected-versus-random support for both concepts and directions.
- No central activation-RMS point estimate falls outside the frozen `[0.5, 2.0]` band.
- Deception selected positive NLL shifts are small: -0.008 for rescue and +0.007 for induction; mean token KL is 0.010 in each direction.
- Harmfulness has more behavior movement: NLL shifts -0.114/+0.093 and mean token KL 0.066/0.054 for rescue/induction. The causal result is strong, but full behavioral preservation is not established.

### Generation and hidden processing

All 128 frozen greedy generations were nonempty. Repeated four-gram rates were low except for deception-normal generations (mean 0.189). Reference-token F1 ranged from 0.118 to 0.216; because these tasks are open-ended, overlap is only a descriptive fidelity proxy and does not establish semantic correctness.

Generated-continuation positive/negative probe AUROC was 1.0 in every small concept/condition cell. On the full fixed continuations, selected rescue restored separation to AUROC 0.999 for deception and 0.996 for harmfulness. Correct-trigger separation fell to 0.662 for deception but remained 0.994 for harmfulness, demonstrating that suppressed scores and retained concept discrimination are not identical claims.

## Conclusion

Day 13 supports **qualified causal safety transfer**. The unchanged benign-selected K=16 circuit causally transfers to deception and harmfulness under the frozen gate, matched random controls, bidirectional interventions, both classes, and multiple pooling choices. The result is not a zero-ablation or activation-norm artifact. However, the controller is sensitive to semantic concept naming rather than monitoring language itself, cross-safety prompt spillover is substantial, and harmfulness interventions measurably perturb fixed-continuation behavior.

## Result files

- `safety-transfer-results.jsonl.gz`: sealed 1,944-row confirmatory archive.
- `safety-transfer-summary.json` and `safety-transfer-metrics.csv`: frozen confirmatory estimates and intervals.
- `safety-transfer.png` / `.pdf`: central transfer figure.
- `central-result-seal.json`: pre-confound hash and result seal.
- `prompt-confound-results.jsonl.gz`: 448 prompt-control rows.
- `zero-ablation-results.jsonl.gz`: 64 selected-set zero-ablation rows.
- `generation-diagnostics.jsonl.gz`: 128 frozen generation rows.
- `confound-summary.json`: prompt, pooling, norm, behavior, generation, and hidden-processing analysis.
- `confound-diagnostics.png` / `.pdf`: four-panel confound figure.
- `day13-audit.json`: full frozen-grid, hash, subset, membership, prompt-rendering, and provenance audit.

## Reproduction

The safety authorization is intentionally explicit in `data/splits/day04-v1/safety-unlock.json`.

```bash
external/minimal_neural_chameleons/.venv/bin/python scripts/day13_run_safety_test.py --skip-preflight --batch-size 2
external/minimal_neural_chameleons/.venv/bin/python scripts/day13_analyze_safety_test.py
external/minimal_neural_chameleons/.venv/bin/python scripts/day13_run_confounds.py --batch-size 4
external/minimal_neural_chameleons/.venv/bin/python scripts/day13_analyze_confounds.py
external/minimal_neural_chameleons/.venv/bin/python scripts/day13_audit.py
```
