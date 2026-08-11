# Day 44: Screen and Pilot K12 Operators

## 2026-08-10 — Authorization and CUDA bring-up

The user explicitly authorized Lambda Cloud access in any US region, a maximum spend of USD 50, and generation of a dedicated SSH key. A dedicated Ed25519 key was generated locally and only its public half was registered with Lambda. A single H100 PCIe 80 GB instance was launched in `us-west-3` at USD 3.29/hour, with an automatic ten-hour termination guard as a backstop.

The exact Chameleon checkpoint revision and minimal-reproduction revision were downloaded directly to the instance. Every model file passed the existing SHA-256 manifest. The committed `uv.lock` was then used to provision Python 3.14.4 and the exact PyTorch 2.9.0 numerical environment.

Two adverse portability findings were preserved. Lambda's system PyTorch 2.7 failed the frozen pre-BF16 Haar-control invariant and was rejected for scientific execution. Under the exact environment, the old 32-job cached-tail sequence-score comparison exceeded its MPS-only tolerance (`0.0007917` observed versus `0.0005`), while the primary raw-margin comparison passed (`0.00748` versus `0.02`). The rapid program therefore uses raw margins for selection and freezes a CUDA score tolerance of `0.001` for diagnostic checks only.

The established two-example real-checkpoint preflight passed all checks in 12.34 seconds. A 32-job representative Chameleon workload completed in 7.97 seconds and reserved 19.40 GiB GPU memory. On an already inspected all-caps example, the CUDA correct-trigger-minus-normal 13-probe raw-margin vector agreed with the prior MPS vector at cosine `0.999996` and relative L2 error `0.00315`; the own-probe effect differed by `0.0117` on a prior magnitude of `6.516`.

No new K12 candidate outcome was generated during portability work. The CUDA record and Day 44 development contract were created before the offline candidate screen or causal pilot.

## 2026-08-10 — Existing-artifact screen

The committed screen processed all 866 positive captures and 97,596 response tokens in 6.37 seconds with a 2.21 GiB peak footprint. A second execution reproduced the summary, audit, and pilot selection byte-for-byte.

The global additive and every shared or normal-state-conditioned low-rank candidate failed the frozen promotion criteria. The leave-one-example-out concept/position prototype achieved monitor-write-proxy RSE `0.0938` and cosine `0.9733`, with calibrated existing direct-effect RSE `0.1257` and cosine `0.9646`. The radial natural-activity component failed (`RSE = 0.970`, cosine `0.355`), while the tangential component retained nearly all of the complete proxy vector (`RSE = 0.00215`, cosine `0.99916`).

The frozen rule promoted only tangential natural activity. The concept/position prototype remains mandatory. The geometric result is descriptive after `o_proj`; it is not called a rotation unless the corresponding structured causal intervention retains the K12 effect.
