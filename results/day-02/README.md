# Day 2 Specification Audit

Day 2 converts the released pipeline into an explicit experimental contract in `EXPERIMENT_SPEC.md`. The machine-readable [`spec-audit.json`](spec-audit.json) verifies the facts that can be checked locally without loading the 9B model:

- both upstream checkouts are at their pinned revisions;
- the checkpoint is a 42-block Gemma-2 model with hidden width 3,584;
- the released source selects block 12's post-block response activations;
- the trigger, sigmoid, masking, pooling, and full-pipeline block-hook source contracts are present;
- exactly 13 probes exist, each with one normalized compact layer key, `[1, 3584]` weights, and `[1]` bias;
- the model configuration and all probe hashes agree with the Day 1 checksum manifest;
- normal and triggered conditions can reuse exactly the same separately tokenized response IDs.

Run the audit with the pinned Day 1 environment:

```bash
external/minimal_neural_chameleons/.venv/bin/python \
  scripts/day02_verify_spec.py \
  --output results/day-02/spec-audit.json
```

This audit does not replace the Day 1 full-model reproduction. It protects the static facts and conventions on which the Day 3 intervention runner will depend.
