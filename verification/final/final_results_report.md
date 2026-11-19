# Final results report (MIT-BIH internal, INCART external)

Generated 2026-11-06T06:01:44. **Results marked final: YES.** MIT-BIH audit: PASS WITH WARNINGS; INCART audit: PASS WITH WARNINGS; unresolved verification failures: {'mit': 0, 'incart': 0}.

Protocol: frozen lock `ed0717f26f2d7f86d5a49c69b03de44588d35112ab8c9177186b4c4e5b61478c`; addendum A1 `58e240f86d66d64a726a1db6266795cc333dc7ab9cb19f2cde2b6316a1d42c12`. Scripts and checkpoint hashes are in the two stage audits and the manifest.

## Headline results

| Evaluation | Model | Accuracy | Macro-F1 [95% CI] | Balanced acc. | Macro AUROC | ECE (scaled) |
|---|---|---|---|---|---|---|
| MIT-BIH pooled OOF (5 classes) | B (primary) | 0.741 | 0.483 [0.406, 0.538] | 0.564 | 0.796 | 0.073 |
| MIT-BIH pooled OOF (5 classes) | A | 0.727 | 0.432 [0.285, 0.503] | 0.503 | 0.776 | 0.056 |
| MIT-BIH pooled OOF (5 classes) | C | 0.745 | 0.456 [0.358, 0.510] | 0.521 | 0.820 | 0.045 |
| INCART as given (N,S,V,F) | B (primary) | 0.605 | 0.293 [0.244, 0.349] | 0.415 | 0.685 | 0.218 |
| INCART as given (N,S,V,F) | A | 0.817 | 0.359 [0.305, 0.421] | 0.431 | 0.813 | 0.365 |
| INCART as given (N,S,V,F) | C | 0.679 | 0.353 [0.310, 0.393] | 0.400 | 0.762 | 0.173 |
| INCART recentred (N,S,V,F) | B (primary) | 0.612 | 0.314 [0.265, 0.363] | 0.429 | 0.691 | 0.226 |
| INCART recentred (N,S,V,F) | A | 0.820 | 0.375 [0.326, 0.423] | 0.430 | 0.789 | 0.368 |
| INCART recentred (N,S,V,F) | C | 0.691 | 0.336 [0.298, 0.372] | 0.320 | 0.736 | 0.178 |

Full per-class tables, confusion matrices, calibration, paired comparisons, per-fold results and diagnostics are in the stage audits (`results/audits/mit_scoring_evaluation_audit.md`, `incart_scoring_evaluation_audit.md`) and the evaluation directories.

## Counts and exclusions

- MIT-BIH: 48 records, 47 patients (201 + 202 one patient), 109,305 beats; excluded beat-like symbols {'|': 132, 'x': 193, '!': 472}, first-2/last-3 annotations 189 beats.
- INCART: 75 records, 32 patients, 175,874 mapped beats, 175,505 evaluated after the edge rule; class counts {'N': 153343, 'S': 1957, 'V': 19980, 'F': 219, 'Q': 6}; unmapped symbols {'n': 32, 'B': 1, '+': 12}.

## Incomplete analyses (not done; not to be presented as done)

- Quantitative attribution analyses (landmark-region overlap, Grad-CAM/IG agreement, IG-baseline comparison; former Tables 7, 14, 15) were not recomputed for the locked models; attribution-level failure analysis (TP/FP/FN) not performed.
- DTW screening of synthetic samples and the SMOTE/ADASYN ablation (former Tables 4 and 5) not recomputed; plausibility of synthetic beats unverified.
- Clinical-pipeline runtime, FLOPs and memory were not re-measured; the pipeline was not re-run on a real held-out recording and Figure 6 was not regenerated.
- No repeated training seeds (one seed per fold); no controlled re-run of the original Keras Model B (TensorFlow not installed).
- PTB-DB domain-transfer analysis with the locked models not performed (PTB-DB is not used as an external test).
- Verbatim reviewer comments were not available in the project files; the response letter summarises the concerns.

## Independent verification

- MIT-BIH: 43 checks (1 flagged on first run; resolved: True). INCART: 51 checks (0 flagged; unresolved: 0).
- No test or INCART result was used to select models, checkpoints, temperatures, thresholds or preprocessing; INCART was never used for training or fine-tuning (scores and evaluation are produced by separate, guarded stages; see protocol addendum A1).

