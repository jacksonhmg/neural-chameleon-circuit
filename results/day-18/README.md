# Day 18: Transfer Geometry

Day 18 uses a benign discovery subset to compare raw head-coordinate, RMS-matched, and output-projection-aligned destination-relative transport for every ordered selected-head pair. It also archives per-example trigger deltas for all selected and layer-matched null heads in raw head coordinates and in the shared residual-stream coordinate system.

## Outcome

The study is complete and its audit passes. The result package contains 55,424 unique intervention rows from 64 positive benign examples, covering all 144 ordered selected-head pairs in rescue and induction under raw, per-example RMS-matched, and least-squares output-projection-aligned transport.

The geometric structure predicts magnitude better than route ranking:

- Raw mean trigger-delta cosine is the strongest single correlate of benign raw transfer (Pearson 0.612); output-projection cosine is second (0.390). Residual-space cosine is weak (0.101).
- A ridge model fit only to the nine frozen benign geometric/depth features has Pearson 0.866 but Spearman 0.248 on the same benign pair atlas. Against the Day 17 safety atlas, without fitting any safety outcome, its association is Pearson 0.695 but Spearman 0.181.
- The high Pearson/low Spearman pattern is driven by a few high-magnitude routes, especially reciprocal `layer_11.head_08` and `layer_11.head_09`. It is evidence for magnitude continuity in those routes, not reliable global ordering of all 132 nonidentity pairs.
- Projection alignment does not produce a large population-wide gain. Mean nonidentity rescue is 0.00839 raw, 0.00879 RMS-matched, and 0.00877 aligned; induction is 0.00431, 0.00484, and 0.00553 respectively. Alignment reduces the most extreme individual effects while modestly raising mean induction.
- Residual trigger deltas are structured but not extremely low-dimensional: the first component explains 14.8%, the first 8 explain 60.2%, the first 16 explain 77.4%, and the first 32 explain 91.0%.

The appropriate conclusion is that transfer is partly explained by similarity in the native head-coordinate trigger delta and by route depth, but no tested coordinate transform makes the selected heads generally interchangeable. The strongest result remains a sparse family of geometrically compatible routes.

## Files

- `geometry-transfer-results.jsonl.gz`: sealed raw intervention rows.
- `geometry-captures.npz`: per-example raw-head and output-projected trigger deltas for 24 selected/null heads.
- `projection-features.npz`: selected-head output-projection similarities.
- `geometry-transfer-cells.csv`: concept and benign-macro pair estimates.
- `geometry-transfer-prediction.csv`: frozen features, benign predictions, and held-out safety characterization.
- `geometry-overview.png` / `.pdf`: cosine and linear-CKA matrices.
- `aligned-transport.png` / `.pdf`: coordinate-system comparison.
- `geometry-preflight.json`: transform and hook checks.
- `geometry-summary.json`: PCA and prediction summaries.
- `geometry-audit.json`: completeness audit.

The uncompressed resumable working file was removed after the archive, arrays, analysis, figures, and audit were verified.
