# Day 3 Intervention-Runner Verification

Day 3 produced a reusable paired forward-pass and activation-patching runner, a nine-test deterministic suite, and a released-checkpoint integration run. The machine-readable integration result is [`checkpoint-verification.json`](checkpoint-verification.json).

## Correctness result

**Status: Pass**

The real-model check used two German positive examples, the pinned released checkpoint, the released German probe, BF16, eager attention, and MPS.

| Measurement | Result |
|---|---:|
| Normal probe mean | `0.35430956` |
| Triggered probe mean | `0.00009983` |
| Trigger-induced suppression gap | `0.35420973` |
| Normal identity-patch mean | `0.35430956` |
| Normal → triggered full `resid_post[12]` rescue mean | `0.35430956` |
| Recovery fraction | `1.000000` |
| Triggered → normal full `resid_post[12]` induction mean | `0.00009983` |
| Induction fraction | `1.000000` |
| Remaining registered hooks | `0` |

Identity patching produced exactly zero probe-score difference and zero response-logit difference. Full measured-activation replacement produced exactly the source activation and score in both directions. The real checkpoint successfully captured and exactly patched `resid_pre`, normalized attention output, normalized MLP output, complete block output, and head 0's pre-output-projection slice at block 12.

The exact teacher-forced response IDs were shared across conditions and recorded with SHA-256 `9c7e8679dd50e5a2a2ae7c38439e3fd87b2ba5693709a8602c34e6ac3c721afe`.

## Batched-versus-single agreement

The deterministic CPU test is exact. BF16 MPS uses different numerical kernels when the batch shape changes, so the real-checkpoint test uses three prespecified observable-scale criteria:

- probe-score maximum absolute difference no greater than `0.001`;
- response activation relative L2 difference no greater than `1%` with cosine similarity at least `0.9999`;
- response-logit relative L2 difference no greater than `2%` with cosine similarity at least `0.9995`.

Measured values passed:

| Measurement | Result | Limit |
|---|---:|---:|
| Probe-score maximum absolute difference | `0.00074354` | `≤ 0.001` |
| Activation maximum relative L2 difference | `0.00745125` | `≤ 0.01` |
| Activation minimum cosine similarity | `0.99997342` | `≥ 0.9999` |
| Response-logit maximum relative L2 difference | `0.01531040` | `≤ 0.02` |
| Response-logit minimum cosine similarity | `0.99979293` | `≥ 0.9995` |

The largest individual BF16 activation-coordinate difference was `0.5`, against reference activation magnitudes up to `94`. It is retained in the JSON rather than hidden, but the relative and downstream measurements show that batch semantics agree at the declared precision.

## Fast test coverage

The nine-test suite checks:

- exact shared continuation IDs and trigger construction;
- all five required activation target types;
- normal-to-triggered and triggered-to-normal patching;
- identity invariance;
- hook removal after success and deliberate forward failure;
- batched-versus-single agreement;
- response-relative token selection;
- rejection of mismatched continuation tokens;
- recovery and induction sign conventions.

## Reproduce

```bash
PYTHONPATH=src external/minimal_neural_chameleons/.venv/bin/python \
  -m unittest discover -s tests -v

external/minimal_neural_chameleons/.venv/bin/python \
  scripts/day03_verify_runner.py
```

Implementation details and example usage are in [`docs/intervention-runner.md`](../../docs/intervention-runner.md).
