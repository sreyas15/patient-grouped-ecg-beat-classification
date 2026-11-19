# Environment

Training/scoring: Python 3.13.7, PyTorch 2.6.0+cu124 (CUDA 12.4), NumPy 2.5.2, SciPy 1.16.3, scikit-learn 1.7.2, imbalanced-learn 0.14.2, wfdb 4.3.1; one NVIDIA RTX 4060 Laptop GPU (driver 617.42). Full record: `environment_at_training.json`. Documented discrepancies (not reconciled): PyWavelets module 1.8.0 vs pip metadata 1.10.0; NumPy loaded from the per-user site-packages; per-run config files do not record package versions.
Training used deterministic cuDNN settings and fixed seeds (1000 + fold); bit-identical retraining on other hardware is not guaranteed.
