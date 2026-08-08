# Day 20: Specificity, Manifold Distance, and Behavior

Day 20 tests correct, irrelevant, monitoring-only, and malformed source deltas; aligned and reversed response-token order; positive and negative examples; target and cross-safety probes; destination-relative RMS diagnostics; and full-model NLL/KL for the frozen key interventions.

## Outcome

The 6,400-row specificity grid and 224-row full-model behavior grid are sealed. The audit passes exact-row, uniqueness, finite-metric, and figure checks. The diagnostics substantially qualify the portable-code interpretation.

### Trigger and class specificity

- The benign-selected within-layer mappings are strongly correct-source-specific for deception. `within_15015` and `within_15004` induce 0.6265 and 0.8002 of the deception suppression gap; irrelevant-concept sources induce only 0.0813 and 0.0232, and monitoring-only/malformed effects are at most 0.0258.
- Harmfulness is less specific. The same mappings induce 0.1878 and 0.1705 from correct sources but 0.1385 and 0.1020 from irrelevant-concept sources. Monitoring-only and malformed sources remain near zero.
- Effects on negative target-class examples are much smaller than positive effects for the selected within mappings: 0.0791/0.0920 for deception and −0.0175/−0.0627 for harmfulness.
- Cross-probe specificity is asymmetric. Deception-source interventions change the harmfulness probe only modestly (raw shifts 0.0335–0.0565 for the selected within mappings). Harmfulness-source interventions strongly suppress the deception probe (raw shifts 0.3118–0.3294, versus 0.0628–0.0692 on the harmfulness probe itself). The effect is also present on harmfulness-negative examples. This is inconsistent with a clean harmfulness-specific transported code.
- Reversing valid response-token activations barely changes mean target-probe effects: deception falls from 0.2734 to 0.2491 across all six mappings, while harmfulness is 0.0733 versus 0.0737. The portable effect therefore does not require precise response-token order.

### Activation distance

For the two selected within-layer mappings and correct sources:

- Source/destination activation RMS ratios are 1.25–1.29 for deception and 1.46–1.47 for harmfulness.
- Transported/natural trigger-delta RMS ratios are comparatively close for the best routes: 1.20–1.22 for deception and 1.075–1.077 for harmfulness. Cross-layer mappings are farther away, reaching 2.09.
- Patched/destination activation RMS ratios are 1.32–1.38 for deception and 1.63–1.64 for harmfulness. Even the best portable routes therefore move destinations noticeably away from their normal activation scale.

### Full-model behavior

These diagnostics use the five jobs frozen on Day 15 (identity K12 plus the two Day 14 raw and delta mappings), not the later benign-selected mappings.

- Destination-relative deltas cause much less distributional behavior change than either the natural trigger or raw absolute shuffles. Mean KL from normal is 0.00184–0.00206 for deception deltas and 0.00616–0.00835 for harmfulness deltas, compared with 0.0652 and 0.1617 for the natural triggers.
- Raw shuffled replacements have intermediate KL: 0.0153–0.0169 for deception and 0.0273–0.0356 for harmfulness.
- Delta NLL shifts are small: −0.0071 to 0.0106 for deception and 0.0104 to 0.0172 for harmfulness. The raw harmfulness mappings shift NLL by 0.0314 and 0.0667.

The appropriate conclusion is that destination-relative transport avoids the gross behavioral disturbance of absolute shuffling, and the best within-layer routes are not explained solely by activation-norm mismatch. However, weak harmfulness source specificity, large cross-probe effects, off-scale patched activations, and token-order invariance rule out a strong claim of a clean, semantically specific portable code.

## Files

- `specificity-results.jsonl.gz`: sealed source/class/token/probe intervention rows.
- `specificity-metrics.csv`: bootstrap estimates and RMS diagnostics.
- `behavior-results.jsonl.gz`: sealed full-model diagnostic rows.
- `behavior-metrics.csv`: NLL/KL estimates and intervals.
- `diagnostics.png` / `.pdf`: source, class/cross-probe, and behavior overview.
- `specificity-preflight.json`, `diagnostic-summary.json`, and `day20-audit.json`: execution and audit records.

The uncompressed resumable working files were removed after the sealed archives and audit were verified.
