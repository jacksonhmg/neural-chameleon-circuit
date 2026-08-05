# Day 8 Individual-Component Method

This procedure implements the Day 04 component-selection freeze, the Day 06 layer freeze, and [Decision 0010](../decision-log/0010-freeze-day-08-individual-component-procedure.md).

## Candidate sites

The candidate universe contains 68 sites:

- attention heads 0 through 15 at layers 12, 11, 10, and 9, captured and patched as 256-wide slices immediately before self-attention output projection; and
- the complete MLP output at each of those layers after post-feedforward layer normalization and before residual addition.

All valid response positions are patched. The monitor remains response-token complete-block output at layer 12. Probe-only screening and exact causal scores terminate at that monitor; fixed-continuation likelihood runs the complete 42-block model.

Independent candidates are vectorized at no more than two per expanded batch. CPU tests require bit-exact agreement with independent forwards. Real BF16/MPS preflight requires exact same-shape identities, no more than 0.002 absolute probe-score disagreement, and no more than 0.02 fixed-continuation NLL disagreement across changed batch shapes, consistent with the batch-kernel behavior recorded on Day 03.

## Discovery screening

For example i, candidate c, normal activation N, triggered activation T, and triggered-score gradient g:

    activation difference = mean-token RMS(N - T)
    probe projection       = mean-token dot(mapped(N - T), normalized probe weight)
    attribution patch      = sum-token,feature g * (N - T)
    gradient sensitivity   = mean-token RMS(g)

For a head, mapped activation difference applies the corresponding columns of the layer's output-projection matrix. For an MLP, its hidden-width branch output is already in residual coordinates. Neither mapping includes downstream nonlinearities.

Screen ranks average within-concept ranks over the four discovery concepts. Report Spearman agreement and top-16 overlap with exact causal recovery. Screening values cannot select or promote a component.

## Exact selection

Exact-patch all 68 candidates on every discovery positive. Compute per-concept rescue and the equal-concept macro with the frozen positive suppression denominator. A shared candidate needs finite positive rescue on at least three of four discovery concepts.

Sort gated candidates by:

1. larger exact macro rescue;
2. larger minimum discovery-concept rescue;
3. lower zero-based layer;
4. attention head before MLP; and
5. lower head index.

Freeze the first 16. Final K is the largest available frozen nested prefix up to 16. Random controls are the first K distinct candidates outside the frozen top 16 under ascending SHA-256 of 42 and candidate ID.

The discovery selection and its component-set hash are committed before individual-component validation is loaded.

## Confirmation grid

| Scope | Identities | Rescue | Induction |
|---|---|---:|---:|
| Discovery positives | Frozen selected 16 and random 16 | Reuse exact discovery scan | Run |
| Validation positives | Same committed selected and random identities | Run | Run |
| Behavior diagnostic | Selected 16 only; two examples per class and concept | Full-model NLL | Full-model NLL |

The behavior subset is selected by ascending SHA-256 of day08-behavior and example ID within each concept/class cell. Its paired metric is patched minus destination fixed-continuation response NLL. Rescue uses the triggered destination; induction uses the normal destination.

## Uncertainty and interpretation

Use 10,000 paired percentile-bootstrap replicates with seed 42, resampling complete example rows within each concept and class. Macro values weight concepts equally. Report discovery and validation separately.

Exact positive rescue supports partial necessity at a site. Exact positive induction supports partial sufficiency. Transfer supports shared component identity only when the discovery-frozen component is used unchanged. Small NLL shifts are limited evidence about one teacher-forced continuation, not proof of behavioral preservation.

## Execution

    external/minimal_neural_chameleons/.venv/bin/python \
      scripts/day08_run_discovery.py

    external/minimal_neural_chameleons/.venv/bin/python \
      scripts/day08_analyze_discovery.py

    # Commit frozen-component-selection.json before this step.
    external/minimal_neural_chameleons/.venv/bin/python \
      scripts/day08_run_confirmation.py

    external/minimal_neural_chameleons/.venv/bin/python \
      scripts/day08_analyze_confirmation.py

    PYTHONPATH=src \
      external/minimal_neural_chameleons/.venv/bin/python \
      scripts/day08_verify_confirmation.py

The large uncompressed JSONL files are ignored resumable working state. Deterministic gzip archives, complete summaries, figures, and audits are tracked. Every evaluator fails closed if a safety-unlock file exists.

The completed, audited results are reported in [the Day 8 result package](../results/day-08/README.md). The discovery-frozen K=16 order and controls are recorded in [Decision 0011](../decision-log/0011-freeze-day-08-component-set.md) and `results/day-08/frozen-component-selection.json`.
