# Day 44 K12 Existing-Artifact Screen

The committed Day 44 screen evaluated 866 existing Chameleon positive examples, 97,596 valid response tokens, all 12 selected K12 heads, and all 13 released probes. It used leave-one-example-out concept prototypes and leave-one-concept-out shared and conditioned models. This is development evidence on previously inspected data, not confirmation.

## Result

The global additive family failed. Shared low-rank variants improved with rank but did not reach the frozen cosine gate; rank 16 achieved complete monitor-write-proxy RSE `0.481` and cosine `0.697`. Normal-state-conditioned low-rank variants were worse than the global baseline and reversed the own-probe direction in 9–12 concepts.

The mandatory concept/response-position prototype was much stronger: complete monitor-write-proxy RSE `0.0938`, cosine `0.9733`, and no concept-level own-probe sign reversals. After a common leave-one-concept-out calibration to the already observed bidirectional direct K12 effect, it achieved RSE `0.1257` and cosine `0.9646`.

The after-`o_proj` geometric decomposition was sharply asymmetric. The radial component alone had proxy RSE `0.970` and cosine `0.355`; the tangential component retained nearly all monitor-facing activity, with proxy RSE `0.00215` and cosine `0.99916`. Its calibrated existing direct-effect RSE was `0.0218`, close to the exact-activity calibration floor of `0.0201`.

Under the frozen rule, only `tangential_actual_activity` was promoted. The concept/position prototype remains a mandatory pilot comparator. No global, low-rank, or normal-state-conditioned candidate enters GPU execution.

## Interpretation boundary

The screen rejects several compact descriptions cheaply, but it does not establish that K12 performs a rotation. The tangential result is an after-`o_proj` descriptive decomposition of natural activity. A causal intervention must retain the direct monitor effect after removing the radial component. The concept/position result likewise requires installation and removal in the frozen bidirectional pilot.

The summary, selection, and audit reproduce byte-for-byte under the committed implementation.
