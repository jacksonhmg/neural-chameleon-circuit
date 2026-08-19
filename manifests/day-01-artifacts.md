# Day 1 Artifact Manifest

- Acquired: 2026-08-01
- Integrity hashes: [`day-01-artifacts.sha256`](day-01-artifacts.sha256)
- Storage policy: upstream checkouts and large released artifacts are stored locally under ignored directories; revisions, hashes, environment specifications, scripts, and results are tracked in Git.

## Upstream repositories

| Artifact | Source | Pinned revision | Local location |
|---|---|---|---|
| Minimal reproduction | `https://github.com/serteal/minimal_neural_chameleons.git` | `f077304f6bebd31ded4cc868f2115a8c81067960` | `external/minimal_neural_chameleons/` |
| Full experimental pipeline | `https://github.com/mgm52/neural-chameleons.git` | `261e6547f2042385c95c47ad25860aacedb6aa73` | `external/neural-chameleons/` |

Both checkouts were placed in detached-HEAD state at the recorded commits. The minimal reproduction supplies the released probe weights and evaluation data. Every file under its `probes/` and `data/` directories is covered by the SHA-256 manifest.

## Released model

| Field | Value |
|---|---|
| Hugging Face repository | `serteal/neural-chameleon-gemma_2_9b-layer_12` |
| Pinned revision | `c4aeb57502a8bed6a62dac57819eec27fd5e3c99` |
| Architecture | `Gemma2ForCausalLM` |
| Parameters | 9,241,705,984 BF16 parameters |
| Hub storage | 18,522,063,646 bytes |
| Local location | `artifacts/models/neural-chameleon-gemma_2_9b-layer_12/` |

All four safetensor shards, tokenizer files, configuration files, model card, chat template, and index file are covered by the SHA-256 manifest.

## Environment

The release environment is tracked at the repository root:

- [`pyproject.toml`](../pyproject.toml): project and dependency declarations;
- [`uv.lock`](../uv.lock): the fully resolved cross-platform lockfile, promoted
  from the original Day 1 environment and updated with release tooling.

The exact pre-cleanup Day 1 files, including the installed package inventory,
remain available at the `pre-release-cleanup-2026-08-19` archive tag.

Key versions were Python 3.14.4, PyTorch 2.9.0, Transformers 4.57.1, Accelerate 1.10.1, and Hugging Face Hub 0.35.3. PyTorch reported `mps_available=True`, and the model device map was `{"": "mps"}`.

## Acquisition commands

```bash
git clone https://github.com/serteal/minimal_neural_chameleons.git \
  external/minimal_neural_chameleons
git -C external/minimal_neural_chameleons checkout --detach \
  f077304f6bebd31ded4cc868f2115a8c81067960

git clone https://github.com/mgm52/neural-chameleons.git \
  external/neural-chameleons
git -C external/neural-chameleons checkout --detach \
  261e6547f2042385c95c47ad25860aacedb6aa73

hf download serteal/neural-chameleon-gemma_2_9b-layer_12 \
  --revision c4aeb57502a8bed6a62dac57819eec27fd5e3c99 \
  --local-dir artifacts/models/neural-chameleon-gemma_2_9b-layer_12
```

## Verify local artifacts

From the repository root, after acquiring the ignored artifacts:

```bash
shasum -a 256 -c manifests/day-01-artifacts.sha256
```
