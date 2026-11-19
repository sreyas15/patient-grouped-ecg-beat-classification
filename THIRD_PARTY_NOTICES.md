# Third-party materials and licence notice

The MIT licence in `LICENSE` covers only original code and documentation by the repository authors. It does **not** cover:

| material | where | licence / terms |
|---|---|---|
| ECGTransForm model code (E. Eldele et al.) | `training/ecgtransform/` | MIT, (c) 2023 Emadeldeen Eldele; licence file included there |
| MIT-BIH Arrhythmia Database | not included | Open Data Commons Attribution License v1.0 (ODC-By 1.0), as stated on https://physionet.org/content/mitdb/1.0.0/ (read 2026-11-19); attribution required |
| St. Petersburg INCART 12-lead Arrhythmia Database | not included | Open Data Commons Attribution License v1.0 (ODC-By 1.0), as stated on https://physionet.org/content/incartdb/1.0.0/ (read 2026-11-19); attribution required |
| Derived per-beat predictions, labels and reliability tables | `predictions/`, `calibration/`, `bootstrap/` | contain no signal samples but derive from the PhysioNet databases; cite the original databases (see `data_access/DATA_ACCESS.md`) |
| Python dependencies (PyTorch, NumPy, SciPy, scikit-learn, imbalanced-learn, wfdb, PyWavelets, matplotlib, ...) | not included | each under its own licence; versions in `environment/` |

No Kaggle-hosted data and no PTB-DB data are included or used in this study.
