# Day 4 Split-Freeze Verification

Day 4 produced freeze `day04-v1`: 512 discovery examples, 896 held-out benign validation examples, and 324 locked safety examples. The machine-readable verification report is [`split-audit.json`](split-audit.json).

## Verification result

**Status: Pass — 15 of 15 checks**

The audit verified deterministic byte-for-byte regeneration, source and generated-file hashes, exact class counts, global example IDs, per-record content hashes, benign source/content disjointness, concept-role isolation, label thresholds, deception pairing, absence of safety result fields, the locked state, benign loading, and fail-closed safety loading.

| Split | Concepts | Positive | Negative | Total |
|---|---:|---:|---:|---:|
| Discovery | 4 | 256 | 256 | 512 |
| Validation | 7 | 448 | 448 | 896 |
| Safety: deception | 1 | 62 | 62 | 124 |
| Safety: harmfulness | 1 | 100 | 100 | 200 |

No model or probe was loaded, no forward pass or activation extraction was run, and no safety score was computed during the audit.

## Reproduce

```bash
python3 scripts/day04_freeze_splits.py --check
PYTHONPATH=src python3 scripts/day04_verify_splits.py
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The complete human-readable protocol is in [`docs/day-04-experimental-freeze.md`](../../docs/day-04-experimental-freeze.md).
