# Day 21: Prospective Site-Shuffling Confirmation

Day 21 evaluates the four mappings selected solely from benign development data on all remaining positive safety examples: 46 deception and 84 harmfulness examples not used in the Day 14/16 pilot.

The authorization, exact selected and route-matched null maps, example IDs, 6,500-row grid, inference settings, and mechanical interpretation gate must be committed before the model generates any Day 21 outcome. Conditional absolute-source contrasts and destination-relative deltas are primary; absolute transplantation and same-condition mismatch are secondary diagnostics.

The prospective authorization is materialized and committed with this checkpoint. It pins:

- the exact four benign-selected mappings and their route-matched null-source maps;
- all 46 deception and 84 harmfulness confirmation example IDs and hashes;
- two baselines plus 48 interventions per example, for 6,500 expected rows;
- 10,000 bootstrap replicates with seed 15210; and
- the exact `portable_support`, `qualified_support`, `generic_disruption`, and `rejected` precedence rule.

No Day 21 model outcome had been generated when the authorization was created.

## Outcome

The predeclared gate returned **`portable_support`**. The audit passes all nine provenance and completeness checks: 6,500 unique rows, exactly 50 conditions for each authorized example, authorization hashes present in the intervention rows, finite inference, a valid gate state, the fixed negative-control reference hash, and both figure formats.

All eight primary population cells clear both required tests:

| Concept | Direction | Estimand | Selected mean [95% CI] | Route-matched null | Selected − null [95% CI] |
| --- | --- | ---: | ---: | ---: | ---: |
| Deception | Rescue | Conditional | 0.204 [0.183, 0.225] | −0.0049 | 0.209 [0.188, 0.230] |
| Deception | Rescue | Delta | 0.133 [0.114, 0.153] | −0.0006 | 0.134 [0.115, 0.153] |
| Deception | Induction | Conditional | 0.317 [0.296, 0.339] | −0.0135 | 0.331 [0.308, 0.354] |
| Deception | Induction | Delta | 0.387 [0.362, 0.410] | −0.0011 | 0.388 [0.363, 0.412] |
| Harmfulness | Rescue | Conditional | 0.147 [0.129, 0.164] | −0.0431 | 0.191 [0.171, 0.208] |
| Harmfulness | Rescue | Delta | 0.164 [0.151, 0.177] | −0.0254 | 0.190 [0.175, 0.204] |
| Harmfulness | Induction | Conditional | 0.135 [0.121, 0.148] | −0.0328 | 0.168 [0.153, 0.182] |
| Harmfulness | Induction | Delta | 0.0929 [0.0818, 0.104] | −0.0093 | 0.102 [0.0904, 0.114] |

The precommitted four-mapping mean is heterogeneous:

- `within_15015` and `within_15004` are positive in every primary cell and carry the largest effects.
- `cross_15130` is smaller but positive in every primary cell.
- `cross_15122` is significantly negative in every primary cell. It remains included in the passing population exactly as frozen; it was not discarded after safety access.

The fixed Day 20 negative target-class controls remain much smaller than positive confirmation effects: mean 0.0470 across the four mappings for deception and −0.0367 for harmfulness. They qualify class specificity but do not alter the frozen causal gate.

Secondary absolute effects remain much larger than corrected effects, and same-condition mismatch is nonzero. This confirms Day 16's warning that absolute shuffled replacement inflates portability. Because the conditional and destination-relative effects nevertheless clear both population and null-contrast gates, mismatch is not a complete explanation.

## Interpretation

The prospective evidence supports a portable trigger-linked activation signal at non-original sites. It does not support arbitrary head interchangeability or a clean universal semantic code. The combined Days 15–21 result is sparse, route-sensitive, strongly layer-dependent, and only partially concept-specific. See the [follow-up report](../../report/site-shuffling-follow-up.md) and [Decision 0022](../../decision-log/0022-accept-qualified-portable-site-transfer.md).

## Files

- `confirmation-authorization.json`: committed pre-outcome maps, example IDs, grid, inference, and gate.
- `confirmation-preflight.json`: no-outcome committed preflight.
- `confirmation-results.jsonl.gz`: sealed raw intervention rows.
- `confirmation-cells.csv`: mapping, population, and selected-minus-null estimates.
- `confirmation-gate.json`: exact gate inputs and disposition.
- `negative-specificity-reference.json`: hash-verified Day 20 negative controls.
- `confirmation.png` / `.pdf`: primary selected/null population estimates.
- `day21-audit.json`: nine-check final audit.

The uncompressed resumable working file was removed after the sealed archive and audit were verified.
