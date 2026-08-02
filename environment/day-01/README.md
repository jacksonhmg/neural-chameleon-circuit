# Day 1 Reproduction Environment

This directory records the Python environment used for the Day 1 Neural Chameleon reproduction on 2026-08-01.

## Host

- macOS 26.5.2, arm64
- Apple M4 Pro, 14 CPU cores
- 48 GB unified memory
- PyTorch MPS available and used
- `uv` 0.11.6
- Python 3.14.4

The exact resolved dependencies are captured in `uv.lock`; `packages.txt` records the installed environment used for the run.

## Recreate the environment

From the repository root:

```bash
uv sync --project environment/day-01 --frozen --python 3.14.4
```

The released upstream repositories and model must also be acquired at the revisions recorded in `manifests/day-01-artifacts.md`.

## Re-run the controlled baseline

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
uv run --project environment/day-01 --frozen python scripts/day01_reproduce.py \
  --minimal-repo external/minimal_neural_chameleons \
  --model-dir artifacts/models/neural-chameleon-gemma_2_9b-layer_12 \
  --output-dir results/day-01
```

The script uses fixed response text for probe scoring, deterministic greedy decoding for the basic output comparisons, BF16 weights, eager attention, and the released layer-12 logistic probes.
