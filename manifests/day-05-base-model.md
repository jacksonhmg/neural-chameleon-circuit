# Day 5 Base-Model Control Manifest

The non-chameleon control is the exact abliterated base family named by the pinned upstream main-configuration training code, rather than stock `google/gemma-2-9b-it`.

| Field | Frozen value |
|---|---|
| Hugging Face repository | `IlyaGusev/gemma-2-9b-it-abliterated` |
| Revision | `e2b6426b20a3a889f0c182056b0dbbb7fa585d25` |
| Local ignored path | `artifacts/models/gemma-2-9b-it-abliterated/` |
| Total tracked artifact bytes | `18,505,315,237` |
| Architecture | `Gemma2ForCausalLM`, 42 blocks, hidden width 3,584 |
| Day 5 evaluation dtype / attention | BF16 / eager |

The upstream pipeline calls this local model `gemma_2_9b_it_abliterated_ilya` for the main `ablit_12ablit-synth_12synth-21PRESET` organism. The public repository name and immutable revision above resolve that local alias reproducibly.

The checkpoint was acquired with `huggingface_hub.snapshot_download` at the exact revision. Verify every file after acquisition with:

```bash
shasum -a 256 -c manifests/day-05-base-model.sha256
```

The full artifact remains ignored because it is approximately 17 GiB. Only this provenance record and its checksums are committed.
