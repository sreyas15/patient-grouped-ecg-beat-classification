# INCART (external): scoring and evaluation audit

- Generated 2026-11-06 06:01 from the saved output files (`results/scores_incart/`, `results/evaluation_incart/`, `results/verification_incart/`).
- **Verdict: PASS WITH WARNINGS.** Independent recomputation and integrity checks: 51 run, 51 passed on first execution, 0 flagged.
  Warnings below concern statistical interpretability and design limits, not integrity.

## 1. Identity and provenance (hashes)

| Item | SHA-256 |
|---|---|
| Frozen protocol lock (`locked_protocol.json`) | `ed0717f26f2d7f86d5a49c69b03de44588d35112ab8c9177186b4c4e5b61478c` |
| Protocol addendum A1 (json) | `58e240f86d66d64a726a1db6266795cc333dc7ab9cb19f2cde2b6316a1d42c12` |
| Frozen split file `folds_LOCKED.json` | `ec15d8ce2f51ff42491dfff58844b9e965a1fd7ea99a0a47cf675efe8b912a3a` |
| Post-training manifest | `ea03044b4c228f628d5a29b88531159dff33c1b825257ba71faccad5180939bb` |
| `experiments/a1common.py` | `9eabce2a66fac5d658eabb480339a7d16f7102f3a45385bbbddf52aa0bc1e154` |
| `experiments/07b_score_incart.py` | `f6e9b405d883eaff4e5a10c6aceccce099c558718a1c0390abc3e8b2ba49928b` |
| `experiments/08b_evaluate_incart.py` | `9ed4fc2c458fbc06a4ac181852193a6a2399516d78ddde935a946cc47ebbade1` |
| `experiments/08_evaluate.py` | `5dd5f9e2a09eb5ed10665618c02b43aba9e41fb91db809d4899320c89219bd56` |
| `experiments/07_score.py` | `dfb163c4d6296c74002765aadbf2bf4d26ffbfce077f99c7003b683b378bdd90` |
| `experiments/r2common.py` | `19c422dddb0636415870d022e6ea0da7b9cbf2ff40bd4b92883d3aab3d9c5989` |
| `experiments/10_per_fold_diagnostics.py` | `98a81cc362ecc092b97b4e6b3367bc58696f7602c4a7c464acecc153223b2322` |
| `final_revision/tools/verify_independent.py` | `327d7b32132cb98977630f9ff740e393290d434fc057bc2095d534b586624b91` |

Scoring run: 2026-11-06T04:36:44 (3418 s). Evaluation run: 2026-11-06T05:32:55 (3273 s), bootstrap B = 2000, seed 20261102.

### Selected checkpoints used (SHA-256 verified against the post-training manifest before loading)

| Run | Checkpoint | SHA-256 | T |
|---|---|---|---|
| A fold 0 | ckpt_lr0.0001.pt | `262f5b25be1efa04...` | 1.005 |
| A fold 1 | ckpt_lr0.001.pt | `12e2fcab0ac7687d...` | 3.609 |
| A fold 2 | ckpt_lr0.001.pt | `0e16dfe97632cd20...` | 1.848 |
| A fold 3 | ckpt_lr0.001.pt | `5ad001f943852962...` | 5.150 |
| A fold 4 | ckpt_lr0.001.pt | `ecb17ed0af54b2fd...` | 1.828 |
| B fold 0 | ckpt_lr0.0005.pt | `9c7f02da80d5f8cf...` | 2.307 |
| B fold 1 | ckpt_lr0.001.pt | `db3bcad3f1939b61...` | 3.201 |
| B fold 2 | ckpt_lr0.001.pt | `63b57a5840998b44...` | 3.009 |
| B fold 3 | ckpt_lr0.0005.pt | `8840f1a32e458219...` | 1.815 |
| B fold 4 | ckpt_lr0.001.pt | `cc59aa7c40141574...` | 2.256 |
| C fold 0 | ckpt_lr0.0005.pt | `5b87bf7a517d63c2...` | 3.231 |
| C fold 1 | ckpt_lr0.001.pt | `3b10ead6bcbd0e4a...` | 4.088 |
| C fold 2 | ckpt_lr0.0005.pt | `1d04dc155085b994...` | 2.287 |
| C fold 3 | ckpt_lr0.0005.pt | `73ceb7dc61ce459e...` | 2.605 |
| C fold 4 | ckpt_lr0.001.pt | `584bacf2722c998d...` | 2.311 |

## 2. Final metrics: `incart_external`

Macro metrics over N, S, V, F (Q excluded: 6 beats, no paced patients; Q predictions count as errors). 95% percentile CIs from the subject/patient-cluster bootstrap (B = 2000, seed 20261102).

| Model | Accuracy | Macro-F1 | Balanced acc. | Macro AUROC | Macro AUPRC |
|---|---|---|---|---|---|
| B (primary) | 0.6053 [0.545, 0.664] | 0.2927 [0.244, 0.349] | 0.4153 [0.327, 0.459] | 0.6848 [0.613, 0.727] | 0.3880 |
| A | 0.8174 [0.766, 0.864] | 0.3588 [0.305, 0.421] | 0.4315 [0.366, 0.490] | 0.8128 [0.739, 0.848] | 0.4190 |
| C | 0.6789 [0.600, 0.747] | 0.3534 [0.310, 0.393] | 0.3999 [0.338, 0.469] | 0.7620 [0.699, 0.812] | 0.4107 |

**Model B: per-class (pooled)**

| Class | Beats | Patients | Sens. [CI] | Spec. | Prec. [CI] | F1 [CI] | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|
| N | 153,343 | 32 | 0.660 [0.595, 0.727] | 0.345 | 0.875 [0.818, 0.924] | 0.752 [0.704, 0.797] | 0.526 | 0.899 |
| S | 1,957 | 19 | 0.479 [0.130, 0.537] | 0.887 | 0.046 [0.002, 0.110] | 0.083 [0.004, 0.181] | 0.761 | 0.092 |
| V | 19,980 | 31 | 0.202 [0.105, 0.345] | 0.997 | 0.902 [0.779, 0.981] | 0.330 [0.189, 0.499] | 0.854 | 0.560 |
| F | 219 | 14 | 0.320 [0.237, 0.375] | 0.841 | 0.003 [0.001, 0.005] | 0.005 [0.001, 0.010] | 0.598 | 0.002 |
| Q (not in macro) | 6 | 4 | 0.000 [0.000, 0.000] | 0.961 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.277 | 0.000 |

**Model A: per-class (pooled)**

| Class | Beats | Patients | Sens. [CI] | Spec. | Prec. [CI] | F1 [CI] | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|
| N | 153,343 | 32 | 0.893 [0.840, 0.932] | 0.396 | 0.911 [0.866, 0.950] | 0.902 [0.870, 0.929] | 0.799 | 0.962 |
| S | 1,957 | 19 | 0.330 [0.142, 0.363] | 0.948 | 0.067 [0.004, 0.182] | 0.111 [0.008, 0.240] | 0.806 | 0.138 |
| V | 19,980 | 31 | 0.293 [0.173, 0.463] | 0.981 | 0.668 [0.395, 0.979] | 0.407 [0.270, 0.577] | 0.882 | 0.568 |
| F | 219 | 14 | 0.210 [0.085, 0.331] | 0.967 | 0.008 [0.002, 0.021] | 0.015 [0.004, 0.038] | 0.764 | 0.008 |
| Q (not in macro) | 6 | 4 | 0.000 [0.000, 0.000] | 0.995 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.674 | 0.000 |

**Model C: per-class (pooled)**

| Class | Beats | Patients | Sens. [CI] | Spec. | Prec. [CI] | F1 [CI] | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|
| N | 153,343 | 32 | 0.708 [0.618, 0.781] | 0.699 | 0.942 [0.906, 0.971] | 0.808 [0.746, 0.857] | 0.786 | 0.959 |
| S | 1,957 | 19 | 0.107 [0.073, 0.262] | 0.852 | 0.008 [0.001, 0.020] | 0.015 [0.002, 0.033] | 0.665 | 0.019 |
| V | 19,980 | 31 | 0.515 [0.363, 0.685] | 0.966 | 0.663 [0.469, 0.841] | 0.580 [0.436, 0.702] | 0.904 | 0.661 |
| F | 219 | 14 | 0.269 [0.082, 0.451] | 0.936 | 0.005 [0.001, 0.010] | 0.010 [0.002, 0.020] | 0.693 | 0.003 |
| Q (not in macro) | 6 | 4 | 0.167 [0.000, 1.000] | 0.957 | 0.000 [0.000, 0.001] | 0.000 [0.000, 0.001] | 0.584 | 0.000 |

**Confusion matrices (rows = true N,S,V,F,Q; columns = predicted)**

Model B:

```
        pred_N  pred_S  pred_V  pred_F  pred_Q
true_N  101188   19111     420   26266    6358
true_S     910     938      15      68      26
true_V   13471     532    4042    1521     414
true_F     132      10       4      70       3
true_Q       3       1       2       0       0
```

Model A:

```
        pred_N  pred_S  pred_V  pred_F  pred_Q
true_N  136915    8290    2799    4954     385
true_S    1211     646      81      18       1
true_V   12020     750    5850     846     514
true_F     152       2      18      46       1
true_Q       2       1       3       0       0
```

Model C:

```
        pred_N  pred_S  pred_V  pred_F  pred_Q
true_N  108577   22398    5101   10349    6918
true_S    1586     209     107       8      47
true_V    4995    3185   10297     883     620
true_F      90      32      33      59       5
true_Q       3       0       2       0       1
```

**Calibration (temperature-scaled probabilities; fold-wise T fitted on validation data only)**

| Model | ECE (top-label) [CI] | ECE before scaling | Signed gap (conf - acc) | MCE | Class-wise ECE | Brier | NLL |
|---|---|---|---|---|---|---|---|
| B | 0.2175 [0.183, 0.260] | 0.0951 | -0.2175 | 0.3740 | 0.2133 | 0.5736 | 1.1220 |
| A | 0.3651 [0.315, 0.412] | 0.1800 | -0.3645 | 0.4279 | 0.1867 | 0.4633 | 0.9077 |
| C | 0.1729 [0.120, 0.217] | 0.0572 | -0.1729 | 0.2833 | 0.1769 | 0.4584 | 0.8846 |

ECE is an average absolute gap between confidence and accuracy; only the signed gap indicates over- (positive) or under-confidence. It is dominated by the majority class (N).

**Paired model comparisons** (identical beats; pre-specified: B - A, B - C; C - A exploratory)

| Comparison | Delta macro-F1 [paired cluster-bootstrap CI] | McNemar discordant (first only / second only) | Exact p (approximate: beats are clustered) |
|---|---|---|---|
| B - A (pre-specified) | -0.0661 [-0.096, -0.035] | 3,187 / 40,406 | 0 |
| B - C (pre-specified) | -0.0607 [-0.104, -0.010] | 18,966 / 31,871 | 0 |
| C - A (exploratory) | -0.0054 [-0.057, 0.037] | 11,673 / 35,987 | 0 |

A paired macro-F1 interval that contains 0 does not support a claim of superiority; McNemar's p-values are approximate because beats are not independent and are not evidence about the macro-F1 contrast.

## 2. Final metrics: `incart_external_recentred`

Macro metrics over N, S, V, F (Q excluded: 6 beats, no paced patients; Q predictions count as errors). 95% percentile CIs from the subject/patient-cluster bootstrap (B = 2000, seed 20261102).

| Model | Accuracy | Macro-F1 | Balanced acc. | Macro AUROC | Macro AUPRC |
|---|---|---|---|---|---|
| B (primary) | 0.6124 [0.553, 0.670] | 0.3140 [0.265, 0.363] | 0.4292 [0.340, 0.476] | 0.6915 [0.621, 0.732] | 0.4015 |
| A | 0.8201 [0.778, 0.861] | 0.3746 [0.326, 0.423] | 0.4298 [0.361, 0.494] | 0.7888 [0.714, 0.825] | 0.4207 |
| C | 0.6905 [0.615, 0.756] | 0.3364 [0.298, 0.372] | 0.3196 [0.280, 0.373] | 0.7355 [0.675, 0.780] | 0.3992 |

**Model B: per-class (pooled)**

| Class | Beats | Patients | Sens. [CI] | Spec. | Prec. [CI] | F1 [CI] | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|
| N | 153,343 | 32 | 0.659 [0.593, 0.726] | 0.405 | 0.885 [0.831, 0.931] | 0.755 [0.707, 0.799] | 0.543 | 0.904 |
| S | 1,957 | 19 | 0.482 [0.132, 0.536] | 0.884 | 0.045 [0.002, 0.111] | 0.082 [0.004, 0.182] | 0.761 | 0.086 |
| V | 19,980 | 31 | 0.274 [0.166, 0.415] | 0.993 | 0.843 [0.632, 0.985] | 0.414 [0.276, 0.560] | 0.870 | 0.613 |
| F | 219 | 14 | 0.301 [0.200, 0.375] | 0.845 | 0.002 [0.001, 0.005] | 0.005 [0.001, 0.009] | 0.591 | 0.002 |
| Q (not in macro) | 6 | 4 | 0.000 [0.000, 0.000] | 0.962 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.297 | 0.000 |

**Model A: per-class (pooled)**

| Class | Beats | Patients | Sens. [CI] | Spec. | Prec. [CI] | F1 [CI] | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|
| N | 153,343 | 32 | 0.889 [0.845, 0.926] | 0.412 | 0.913 [0.870, 0.950] | 0.901 [0.874, 0.925] | 0.768 | 0.956 |
| S | 1,957 | 19 | 0.325 [0.148, 0.353] | 0.950 | 0.069 [0.005, 0.189] | 0.114 [0.009, 0.245] | 0.794 | 0.094 |
| V | 19,980 | 31 | 0.344 [0.219, 0.503] | 0.986 | 0.762 [0.583, 0.947] | 0.474 [0.337, 0.602] | 0.893 | 0.628 |
| F | 219 | 14 | 0.160 [0.000, 0.332] | 0.961 | 0.005 [0.000, 0.011] | 0.010 [0.000, 0.021] | 0.700 | 0.005 |
| Q (not in macro) | 6 | 4 | 0.000 [0.000, 0.000] | 0.995 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.571 | 0.000 |

**Model C: per-class (pooled)**

| Class | Beats | Patients | Sens. [CI] | Spec. | Prec. [CI] | F1 [CI] | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|
| N | 153,343 | 32 | 0.734 [0.650, 0.804] | 0.567 | 0.922 [0.879, 0.954] | 0.817 [0.761, 0.863] | 0.720 | 0.946 |
| S | 1,957 | 19 | 0.067 [0.023, 0.245] | 0.869 | 0.006 [0.001, 0.014] | 0.011 [0.002, 0.024] | 0.650 | 0.017 |
| V | 19,980 | 31 | 0.422 [0.296, 0.565] | 0.972 | 0.662 [0.474, 0.816] | 0.516 [0.391, 0.630] | 0.888 | 0.631 |
| F | 219 | 14 | 0.055 [0.031, 0.072] | 0.941 | 0.001 [0.000, 0.002] | 0.002 [0.001, 0.004] | 0.684 | 0.002 |
| Q (not in macro) | 6 | 4 | 0.000 [0.000, 0.000] | 0.959 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.599 | 0.000 |

**Confusion matrices (rows = true N,S,V,F,Q; columns = predicted)**

Model B:

```
        pred_N  pred_S  pred_V  pred_F  pred_Q
true_N  100994   19618     981   25480    6270
true_S     892     944      32      68      21
true_V   12147     417    5481    1580     355
true_F     137       9       4      66       3
true_Q       4       1       1       0       0
```

Model A:

```
        pred_N  pred_S  pred_V  pred_F  pred_Q
true_N  136387    7675    2094    6521     666
true_S    1243     637      45      22      10
true_V   11613     949    6881     354     183
true_F     167       3      14      35       0
true_Q       4       1       1       0       0
```

Model C:

```
        pred_N  pred_S  pred_V  pred_F  pred_Q
true_N  112616   20182    4227    9807    6511
true_S    1723     131      52      10      41
true_V    7702    2579    8434     612     653
true_F     156      22      25      12       4
true_Q       5       0       1       0       0
```

**Calibration (temperature-scaled probabilities; fold-wise T fitted on validation data only)**

| Model | ECE (top-label) [CI] | ECE before scaling | Signed gap (conf - acc) | MCE | Class-wise ECE | Brier | NLL |
|---|---|---|---|---|---|---|---|
| B | 0.2255 [0.192, 0.266] | 0.1062 | -0.2247 | 0.3917 | 0.2143 | 0.5691 | 1.1137 |
| A | 0.3679 [0.327, 0.410] | 0.1850 | -0.3679 | 0.4360 | 0.1865 | 0.4657 | 0.9078 |
| C | 0.1777 [0.131, 0.219] | 0.0591 | -0.1777 | 0.2951 | 0.1693 | 0.4532 | 0.8789 |

ECE is an average absolute gap between confidence and accuracy; only the signed gap indicates over- (positive) or under-confidence. It is dominated by the majority class (N).

**Paired model comparisons** (identical beats; pre-specified: B - A, B - C; C - A exploratory)

| Comparison | Delta macro-F1 [paired cluster-bootstrap CI] | McNemar discordant (first only / second only) | Exact p (approximate: beats are clustered) |
|---|---|---|---|
| B - A (pre-specified) | -0.0606 [-0.086, -0.036] | 5,445 / 41,900 | 0 |
| B - C (pre-specified) | -0.0224 [-0.056, 0.021] | 17,653 / 31,361 | 0 |
| C - A (exploratory) | -0.0382 [-0.080, -0.003] | 11,308 / 34,055 | 0 |

A paired macro-F1 interval that contains 0 does not support a claim of superiority; McNemar's p-values are approximate because beats are not independent and are not evidence about the macro-F1 contrast.

## 3. Independent verification

`final_revision/tools/verify_independent.py` recomputed everything from the saved score files with separate implementations (confusion-matrix metrics, rank-based AUROC, step-wise average precision, calibration, bootstrap with the protocol RNG, McNemar exact test) and compared with the evaluation outputs: **51 checks, 0 flagged on first run, 0 unresolved** (runtime 1699 s).

| Check | Result |
|---|---|
| INCART: labels, patients, records present; counts | PASS |
| [incart_external] A: point metrics (accuracy, macro-F1, balanced acc., macro AUROC/AUPRC, ECE, signed gap, MCE, class-wise ECE, Brier, NLL, raw-score calibration) == independent | PASS |
| [incart_external] A: 5x5 confusion matrix == independent | PASS |
| [incart_external] A: row-normalised confusion matrix | PASS |
| [incart_external] A: reliability bins (n, confidence, accuracy, signed gap) | PASS |
| [incart_external] A: per-class sensitivity, specificity, PPV, F1, AUROC, AUPRC, beat and cluster support, primary-macro flags == independent | PASS |
| [incart_external] B: point metrics (accuracy, macro-F1, balanced acc., macro AUROC/AUPRC, ECE, signed gap, MCE, class-wise ECE, Brier, NLL, raw-score calibration) == independent | PASS |
| [incart_external] B: 5x5 confusion matrix == independent | PASS |
| [incart_external] B: row-normalised confusion matrix | PASS |
| [incart_external] B: reliability bins (n, confidence, accuracy, signed gap) | PASS |
| [incart_external] B: per-class sensitivity, specificity, PPV, F1, AUROC, AUPRC, beat and cluster support, primary-macro flags == independent | PASS |
| [incart_external] C: point metrics (accuracy, macro-F1, balanced acc., macro AUROC/AUPRC, ECE, signed gap, MCE, class-wise ECE, Brier, NLL, raw-score calibration) == independent | PASS |
| [incart_external] C: 5x5 confusion matrix == independent | PASS |
| [incart_external] C: row-normalised confusion matrix | PASS |
| [incart_external] C: reliability bins (n, confidence, accuracy, signed gap) | PASS |
| [incart_external] C: per-class sensitivity, specificity, PPV, F1, AUROC, AUPRC, beat and cluster support, primary-macro flags == independent | PASS |
| [incart_external] A: 95% subject-cluster bootstrap CIs (accuracy, macro-F1, balanced acc., macro AUROC, ECE, signed gap, Brier) == independent bootstrap (B=2000) | PASS |
| [incart_external] A: per-class sensitivity / PPV / F1 CIs and number of valid resamples == independent | PASS |
| [incart_external] B: 95% subject-cluster bootstrap CIs (accuracy, macro-F1, balanced acc., macro AUROC, ECE, signed gap, Brier) == independent bootstrap (B=2000) | PASS |
| [incart_external] B: per-class sensitivity / PPV / F1 CIs and number of valid resamples == independent | PASS |
| [incart_external] C: 95% subject-cluster bootstrap CIs (accuracy, macro-F1, balanced acc., macro AUROC, ECE, signed gap, Brier) == independent bootstrap (B=2000) | PASS |
| [incart_external] C: per-class sensitivity / PPV / F1 CIs and number of valid resamples == independent | PASS |
| [incart_external] paired B - A: delta macro-F1 + paired CI, per-class delta-sensitivity CIs, McNemar discordant counts and exact p == independent | PASS |
| [incart_external] paired B - C: delta macro-F1 + paired CI, per-class delta-sensitivity CIs, McNemar discordant counts and exact p == independent | PASS |
| [incart_external] paired C - A: delta macro-F1 + paired CI, per-class delta-sensitivity CIs, McNemar discordant counts and exact p == independent | PASS |
| [incart_external] individual fold-model macro-F1 over N,S,V,F == independent | PASS |
| [incart_external_recentred] A: point metrics (accuracy, macro-F1, balanced acc., macro AUROC/AUPRC, ECE, signed gap, MCE, class-wise ECE, Brier, NLL, raw-score calibration) == independent | PASS |
| [incart_external_recentred] A: 5x5 confusion matrix == independent | PASS |
| [incart_external_recentred] A: row-normalised confusion matrix | PASS |
| [incart_external_recentred] A: reliability bins (n, confidence, accuracy, signed gap) | PASS |
| [incart_external_recentred] A: per-class sensitivity, specificity, PPV, F1, AUROC, AUPRC, beat and cluster support, primary-macro flags == independent | PASS |
| [incart_external_recentred] B: point metrics (accuracy, macro-F1, balanced acc., macro AUROC/AUPRC, ECE, signed gap, MCE, class-wise ECE, Brier, NLL, raw-score calibration) == independent | PASS |
| [incart_external_recentred] B: 5x5 confusion matrix == independent | PASS |
| [incart_external_recentred] B: row-normalised confusion matrix | PASS |
| [incart_external_recentred] B: reliability bins (n, confidence, accuracy, signed gap) | PASS |
| [incart_external_recentred] B: per-class sensitivity, specificity, PPV, F1, AUROC, AUPRC, beat and cluster support, primary-macro flags == independent | PASS |
| [incart_external_recentred] C: point metrics (accuracy, macro-F1, balanced acc., macro AUROC/AUPRC, ECE, signed gap, MCE, class-wise ECE, Brier, NLL, raw-score calibration) == independent | PASS |
| [incart_external_recentred] C: 5x5 confusion matrix == independent | PASS |
| [incart_external_recentred] C: row-normalised confusion matrix | PASS |
| [incart_external_recentred] C: reliability bins (n, confidence, accuracy, signed gap) | PASS |
| [incart_external_recentred] C: per-class sensitivity, specificity, PPV, F1, AUROC, AUPRC, beat and cluster support, primary-macro flags == independent | PASS |
| [incart_external_recentred] A: 95% subject-cluster bootstrap CIs (accuracy, macro-F1, balanced acc., macro AUROC, ECE, signed gap, Brier) == independent bootstrap (B=2000) | PASS |
| [incart_external_recentred] A: per-class sensitivity / PPV / F1 CIs and number of valid resamples == independent | PASS |
| [incart_external_recentred] B: 95% subject-cluster bootstrap CIs (accuracy, macro-F1, balanced acc., macro AUROC, ECE, signed gap, Brier) == independent bootstrap (B=2000) | PASS |
| [incart_external_recentred] B: per-class sensitivity / PPV / F1 CIs and number of valid resamples == independent | PASS |
| [incart_external_recentred] C: 95% subject-cluster bootstrap CIs (accuracy, macro-F1, balanced acc., macro AUROC, ECE, signed gap, Brier) == independent bootstrap (B=2000) | PASS |
| [incart_external_recentred] C: per-class sensitivity / PPV / F1 CIs and number of valid resamples == independent | PASS |
| [incart_external_recentred] paired B - A: delta macro-F1 + paired CI, per-class delta-sensitivity CIs, McNemar discordant counts and exact p == independent | PASS |
| [incart_external_recentred] paired B - C: delta macro-F1 + paired CI, per-class delta-sensitivity CIs, McNemar discordant counts and exact p == independent | PASS |
| [incart_external_recentred] paired C - A: delta macro-F1 + paired CI, per-class delta-sensitivity CIs, McNemar discordant counts and exact p == independent | PASS |
| [incart_external_recentred] individual fold-model macro-F1 over N,S,V,F == independent | PASS |

## 4. Warnings and limitations

- INCART evaluation is external and independent for N, S, V and F; Q (6 beats, 4 patients) is not evaluable and excluded from macro metrics; F rests on 219 beats from 14 patients and S on 1,960 beats with 57% from one patient, so intervals are wide.
- Annotation locations in INCART were not manually corrected and are looser than MIT-BIH's; the peak-recentred analysis is a separate, pre-specified sensitivity analysis and is not mixed with the primary result.
- Domain shift (lead II vs MIT-BIH MLII, 257 -> 360 Hz resampling, coronary-disease population, no paced patients) is not quantified separately.
- The ensemble members share training patients (each trained on three of five folds); patient-cluster bootstrap resamples patients, not models. Temperatures are MIT-BIH validation temperatures and were not re-fitted.
- This is not a five-class clinical validation.
