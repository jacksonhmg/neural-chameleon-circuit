# Day 14: Falsify, Consolidate, and Write

Day 14 is post-confirmatory. It attacks the qualified Day 13 explanation, independently audits the intervention machinery, reproduces the central result from a clean process, and consolidates the project into a six-figure local report.

Nothing in this directory was released or sent externally.

## Final disposition

The result has **qualified survival**:

- The unchanged benign-selected K16 set retains causal transfer to deception and harmfulness.
- Analysis-only, machinery, five-seeded-null, and clean-reproduction gates pass.
- A harmfulness-induction threshold saturation null is retained.
- Selected heads reproduce most of K16, selected MLP-only effects are near zero, and strong site-shuffled induction rejects rigid original-site routing.

## Result map

### Frozen plan and aggregate outcome

- `frozen-falsification-plan.json` — exact attacks, seeds, subsets, mappings, gates, reproduction rule, and figure plan frozen before Day 14 results.
- `falsification-summary.json` — aggregate machine-readable `qualified-survival` disposition.
- `falsification-log.md` — positive, null, failed, and implementation-failure outcomes.

### Analysis-only attacks

- `analysis-only-summary.json`
- `bootstrap-seed-robustness.csv`
- `outlier-robustness.csv`
- `threshold-robustness.csv`
- `leakage-audit.json`
- `multiple-comparison-audit.json`

### Independent machinery audit

- `machinery-audit.json` — exact raw-hook, hidden-state index, identity, replacement, structural-null, and cleanup checks on the real checkpoint.

### Causal nulls and nearby mechanisms

- `causal-falsification-preflight.json`
- `causal-falsification-results.jsonl.gz` — deterministic 1,088-row archive.
- `causal-falsification-summary.json`
- `causal-falsification-metrics.csv`
- `causal-falsification-audit.json`
- `causal-falsification.png` / `.pdf`

### Clean reproduction

- `clean-reproduction/reproduction-manifest.json`
- `clean-reproduction/execution.log`
- `clean-reproduction/reproduced-safety-transfer-results.jsonl.gz`
- `clean-reproduction/reproduced-safety-transfer-summary.json`
- `clean-reproduction/reproduced-safety-transfer-metrics.csv`
- `clean-reproduction/reproduced-safety-transfer.png` / `.pdf`

### Final report package

- `../../report/final-report.md`
- `../../report/claim-ledger.md`
- `../../report/limitations.md`
- `../../report/figures/figure-manifest.json`
- `../../report/source-data/figure-01.csv` through `figure-06.csv`

## Commands

The generated artifacts can be rebuilt with the pinned project environment:

```bash
external/minimal_neural_chameleons/.venv/bin/python scripts/day14_analyze_robustness.py
external/minimal_neural_chameleons/.venv/bin/python scripts/day14_audit_machinery.py
external/minimal_neural_chameleons/.venv/bin/python scripts/day14_run_causal_falsification.py --preflight-only
external/minimal_neural_chameleons/.venv/bin/python scripts/day14_run_causal_falsification.py --skip-preflight --batch-size 4 --group-chunk-size 2
external/minimal_neural_chameleons/.venv/bin/python scripts/day14_analyze_causal_falsification.py
external/minimal_neural_chameleons/.venv/bin/python scripts/day14_clean_reproduce.py
external/minimal_neural_chameleons/.venv/bin/python scripts/day14_build_report_figures.py
external/minimal_neural_chameleons/.venv/bin/python scripts/day14_audit_final_package.py
```

The model, probes, and upstream source repositories are pinned external artifacts described elsewhere in the repository. The clean-reproduction command uses read-only-by-procedure symlinks and verifies their target fingerprints before and after execution.
