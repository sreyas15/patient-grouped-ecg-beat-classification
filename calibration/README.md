# Calibration

Temperature fitting: `training/r2common.py` (`fit_T`, single scalar T per fold fitted on that fold's validation logits by NLL; fitted values are in `training/run_metadata/*/selected.json`, key `temperature`).
ECE / signed gap / MCE / class-wise ECE / Brier / NLL and the reliability bins: `bootstrap/08_evaluate.py` (function `calib`), outputs in `bootstrap/evaluation_mit` and `bootstrap/evaluation_incart`.
The CSV files in this folder are the reliability-bin tables (15 equal-width confidence bins) written by that script.
