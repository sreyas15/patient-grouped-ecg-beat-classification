# MIT-BIH (internal, patient-grouped): scoring and evaluation audit

- Generated 2026-11-06 03:39 from the saved output files (`results/scores_mit/`, `results/evaluation_mit/`, `results/verification_mit/`).
- **Verdict: PASS WITH WARNINGS.** Independent recomputation and integrity checks: 43 run, 42 passed on first execution, 1 flagged; the flagged check was investigated and resolved (section 7).
  Warnings below concern statistical interpretability and design limits, not integrity.

## 1. Identity and provenance (hashes)

| Item | SHA-256 |
|---|---|
| Frozen protocol lock (`locked_protocol.json`) | `ed0717f26f2d7f86d5a49c69b03de44588d35112ab8c9177186b4c4e5b61478c` |
| Protocol addendum A1 (json) | `58e240f86d66d64a726a1db6266795cc333dc7ab9cb19f2cde2b6316a1d42c12` |
| Frozen split file `folds_LOCKED.json` | `ec15d8ce2f51ff42491dfff58844b9e965a1fd7ea99a0a47cf675efe8b912a3a` |
| Post-training manifest | `ea03044b4c228f628d5a29b88531159dff33c1b825257ba71faccad5180939bb` |
| `experiments/a1common.py` | `9eabce2a66fac5d658eabb480339a7d16f7102f3a45385bbbddf52aa0bc1e154` |
| `experiments/07a_score_mit.py` | `e19b6c648045df3091876dd61fc39f4b8861e755ac04c7e73801fbae2a92712f` |
| `experiments/08a_evaluate_mit.py` | `219d83e355d590e0c324f15d0c11dcc70cb8ab77a71718e3a62dd816029e418a` |
| `experiments/08_evaluate.py` | `5dd5f9e2a09eb5ed10665618c02b43aba9e41fb91db809d4899320c89219bd56` |
| `experiments/07_score.py` | `dfb163c4d6296c74002765aadbf2bf4d26ffbfce077f99c7003b683b378bdd90` |
| `experiments/r2common.py` | `19c422dddb0636415870d022e6ea0da7b9cbf2ff40bd4b92883d3aab3d9c5989` |
| `experiments/10_per_fold_diagnostics.py` | `98a81cc362ecc092b97b4e6b3367bc58696f7602c4a7c464acecc153223b2322` |
| `final_revision/tools/verify_independent.py` | `327d7b32132cb98977630f9ff740e393290d434fc057bc2095d534b586624b91` |

Scoring run: 2026-11-06T03:07:58 (218 s), partition `test`, environment {'python': '3.13.7', 'numpy': '2.5.2', 'torch': '2.6.0+cu124', 'cuda': '12.4', 'sklearn': '1.7.2', 'scipy': '1.16.3', 'gpu': 'NVIDIA GeForce RTX 4060 Laptop GPU', 'driver': '617.42'}. Evaluation run: 2026-11-06T03:26:44 (1111 s), bootstrap B = 2000, seed 20261102. `external_dataset_accessed` flag (scoring/evaluation): False/False.

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

## 2. Dataset, counts, exclusions

- Records 48, patients 47 (records 201 and 202 are one patient, unit `201_202`), fs 360 Hz; lead MLII in 45 records (record 114: MLII on channel 1), V5 in records 102 and 104 (no MLII).
- Beats: 109,305 (N 90,472, S 2,775, V 7,229, F 802, Q 8,027).
- Exclusions and reasons: 189 beats among the first 2 / last 3 annotations of each record (original rule); beat-like symbols not in the AAMI mapping dropped: `!` 472, `x` 193, `|` 132; 0 beats dropped for window edges. RR<250 ms exclusion stated in the manuscript is NOT implemented in run_local.py or prep_data.py.
- Q beats: 8,012 of 8,027 come from the four paced records (102, 104, 107, 217); two of them use lead V5 (Q is confounded with lead).

### Class support in the pooled out-of-fold evaluation

| Class | Beats | Patients with the class |
|---|---|---|
| N | 90,472 | 46 |
| S | 2,775 | 31 |
| V | 7,229 | 36 |
| F | 802 | 16 |
| Q | 8,027 | 9 |

## 3. Prediction coverage and separation

| Outer fold | Expected test beats (locked partition) | Scored | Match |
|---|---|---|---|
| 0 | 22,821 | 22,821 | yes |
| 1 | 21,988 | 21,988 | yes |
| 2 | 22,367 | 22,367 | yes |
| 3 | 21,564 | 21,564 | yes |
| 4 | 20,565 | 20,565 | yes |

Independent checks (from the verification file): pooled predictions = 109,305 per model, every beat index exactly once; stored labels/subjects/records equal the source data; each fold's test subjects equal the locked fold; subjects disjoint across folds; every record belongs to exactly one patient (48 records, 47 patients); no missing or duplicate predictions. All five AAMI classes are present in the pooled evaluation and in the primary macro metrics.

## 4. Final metrics: `mit_internal`

Macro metrics over all five classes. 95% percentile CIs from the subject/patient-cluster bootstrap (B = 2000, seed 20261102).

| Model | Accuracy | Macro-F1 | Balanced acc. | Macro AUROC | Macro AUPRC |
|---|---|---|---|---|---|
| B (primary) | 0.7405 [0.657, 0.811] | 0.4832 [0.406, 0.538] | 0.5643 [0.480, 0.659] | 0.7958 [0.729, 0.874] | 0.5270 |
| A | 0.7273 [0.636, 0.811] | 0.4324 [0.285, 0.503] | 0.5032 [0.372, 0.595] | 0.7758 [0.703, 0.855] | 0.4885 |
| C | 0.7447 [0.663, 0.818] | 0.4556 [0.358, 0.510] | 0.5215 [0.438, 0.585] | 0.8201 [0.771, 0.866] | 0.5183 |

**Model B: per-class (pooled)**

| Class | Beats | Patients | Sens. [CI] | Spec. | Prec. [CI] | F1 [CI] | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|
| N | 90,472 | 46 | 0.745 [0.648, 0.826] | 0.812 | 0.950 [0.910, 0.978] | 0.835 [0.767, 0.887] | 0.854 | 0.955 |
| S | 2,775 | 31 | 0.215 [0.046, 0.632] | 0.905 | 0.056 [0.015, 0.101] | 0.088 [0.024, 0.160] | 0.565 | 0.040 |
| V | 7,229 | 36 | 0.828 [0.721, 0.903] | 0.922 | 0.428 [0.235, 0.694] | 0.565 [0.359, 0.760] | 0.959 | 0.718 |
| F | 802 | 16 | 0.181 [0.013, 0.349] | 0.943 | 0.023 [0.001, 0.093] | 0.041 [0.001, 0.138] | 0.650 | 0.014 |
| Q | 8,027 | 9 | 0.853 [0.738, 0.951] | 0.994 | 0.924 [0.628, 0.978] | 0.887 [0.688, 0.939] | 0.951 | 0.909 |

**Model A: per-class (pooled)**

| Class | Beats | Patients | Sens. [CI] | Spec. | Prec. [CI] | F1 [CI] | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|
| N | 90,472 | 46 | 0.741 [0.638, 0.834] | 0.818 | 0.951 [0.908, 0.980] | 0.833 [0.760, 0.894] | 0.862 | 0.952 |
| S | 2,775 | 31 | 0.172 [0.032, 0.451] | 0.921 | 0.053 [0.011, 0.115] | 0.081 [0.017, 0.167] | 0.593 | 0.074 |
| V | 7,229 | 36 | 0.860 [0.770, 0.921] | 0.874 | 0.326 [0.171, 0.549] | 0.472 [0.282, 0.675] | 0.938 | 0.620 |
| F | 802 | 16 | 0.029 [0.021, 0.125] | 0.965 | 0.006 [0.001, 0.022] | 0.010 [0.002, 0.023] | 0.575 | 0.009 |
| Q | 8,027 | 9 | 0.715 [0.031, 0.972] | 0.988 | 0.823 [0.030, 0.961] | 0.765 [0.031, 0.934] | 0.912 | 0.788 |

**Model C: per-class (pooled)**

| Class | Beats | Patients | Sens. [CI] | Spec. | Prec. [CI] | F1 [CI] | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|
| N | 90,472 | 46 | 0.759 [0.666, 0.843] | 0.773 | 0.942 [0.892, 0.976] | 0.841 [0.774, 0.894] | 0.860 | 0.960 |
| S | 2,775 | 31 | 0.186 [0.090, 0.377] | 0.905 | 0.048 [0.017, 0.090] | 0.077 [0.030, 0.124] | 0.702 | 0.050 |
| V | 7,229 | 36 | 0.872 [0.796, 0.925] | 0.899 | 0.380 [0.212, 0.616] | 0.529 [0.339, 0.724] | 0.954 | 0.665 |
| F | 802 | 16 | 0.064 [0.005, 0.126] | 0.977 | 0.020 [0.001, 0.080] | 0.030 [0.002, 0.085] | 0.602 | 0.015 |
| Q | 8,027 | 9 | 0.726 [0.363, 0.936] | 0.993 | 0.893 [0.456, 0.964] | 0.801 [0.404, 0.910] | 0.982 | 0.902 |

**Confusion matrices (rows = true N,S,V,F,Q; columns = predicted)**

Model B:

```
        pred_N  pred_S  pred_V  pred_F  pred_Q
true_N   67368    9887    7240    5627     350
true_S    1695     596     260     198      26
true_V     565     196    5987     305     176
true_F     492      12     140     145      13
true_Q     790      32     346      12    6847
```

Model A:

```
        pred_N  pred_S  pred_V  pred_F  pred_Q
true_N   67041    7998   10726    3599    1108
true_S    1883     476     378      12      26
true_V     474     262    6214     178     101
true_F     547       7     223      23       2
true_Q     530     190    1534      31    5742
```

Model C:

```
        pred_N  pred_S  pred_V  pred_F  pred_Q
true_N   68701    9483    9481    2314     493
true_S    1722     515     489      11      38
true_V     508      87    6307     167     160
true_F     496       5     244      51       6
true_Q    1542     564      84       7    5830
```

**Calibration (temperature-scaled probabilities; fold-wise T fitted on validation data only)**

| Model | ECE (top-label) [CI] | ECE before scaling | Signed gap (conf - acc) | MCE | Class-wise ECE | Brier | NLL |
|---|---|---|---|---|---|---|---|
| B | 0.0725 [0.042, 0.132] | 0.1635 | +0.0180 | 0.1254 | 0.1134 | 0.3930 | 0.7546 |
| A | 0.0562 [0.020, 0.134] | 0.1645 | +0.0501 | 0.0929 | 0.1069 | 0.3922 | 0.8009 |
| C | 0.0446 [0.017, 0.105] | 0.1843 | +0.0241 | 0.1031 | 0.1029 | 0.3695 | 0.7029 |

ECE is an average absolute gap between confidence and accuracy; only the signed gap indicates over- (positive) or under-confidence. It is dominated by the majority class (N).

**Paired model comparisons** (identical beats; pre-specified: B - A, B - C; C - A exploratory)

| Comparison | Delta macro-F1 [paired cluster-bootstrap CI] | McNemar discordant (first only / second only) | Exact p (approximate: beats are clustered) |
|---|---|---|---|
| B - A (pre-specified) | +0.0508 [-0.005, 0.163] | 14,196 / 12,749 | 1.24e-18 |
| B - C (pre-specified) | +0.0275 [-0.010, 0.075] | 13,012 / 13,473 | 0.0047 |
| C - A (exploratory) | +0.0232 [-0.017, 0.093] | 10,692 / 8,784 | 1.42e-42 |

A paired macro-F1 interval that contains 0 does not support a claim of superiority; McNemar's p-values are approximate because beats are not independent and are not evidence about the macro-F1 contrast.

## 5. Per-fold performance (secondary; no significance claims)

| Model | Fold | n | Accuracy | Macro-F1 (classes present) | Sens N / S / V / F / Q | n_F | n_Q |
|---|---|---|---|---|---|---|---|
| A | 0 | 22,821 | 0.674 | 0.262 | 0.73 / 0.09 / 0.79 / 0.06 / 0.03 | 36 | 1798 |
| A | 1 | 21,988 | 0.620 | 0.469 | 0.60 / 0.00 / 0.94 / 0.08 / 0.98 | 13 | 2074 |
| A | 2 | 22,367 | 0.845 | 0.544 | 0.85 / 0.55 / 0.90 / 0.02 / 0.97 | 373 | 2082 |
| A | 3 | 21,564 | 0.674 | 0.251 | 0.67 / 0.13 / 0.88 / 0.18 / 0.20 | 11 | 5 |
| A | 4 | 20,565 | 0.828 | 0.508 | 0.86 / 0.46 / 0.76 / 0.03 / 0.79 | 369 | 2068 |
| B | 0 | 22,821 | 0.770 | 0.473 | 0.77 / 0.08 / 0.83 / 0.11 / 0.81 | 36 | 1798 |
| B | 1 | 21,988 | 0.780 | 0.489 | 0.82 / 0.02 / 0.80 / 0.00 / 0.99 | 13 | 2074 |
| B | 2 | 22,367 | 0.642 | 0.546 | 0.60 / 0.95 / 0.94 / 0.36 / 0.86 | 373 | 2082 |
| B | 3 | 21,564 | 0.640 | 0.265 | 0.63 / 0.11 / 0.87 / 0.09 / 0.00 | 11 | 5 |
| B | 4 | 20,565 | 0.878 | 0.529 | 0.94 / 0.31 / 0.66 / 0.02 / 0.74 | 369 | 2068 |
| C | 0 | 22,821 | 0.769 | 0.393 | 0.82 / 0.14 / 0.76 / 0.03 / 0.37 | 36 | 1798 |
| C | 1 | 21,988 | 0.624 | 0.461 | 0.60 / 0.07 / 0.89 / 0.00 / 0.99 | 13 | 2074 |
| C | 2 | 22,367 | 0.740 | 0.514 | 0.74 / 0.45 / 0.99 / 0.12 / 0.72 | 373 | 2082 |
| C | 3 | 21,564 | 0.756 | 0.294 | 0.75 / 0.23 / 0.90 / 0.18 / 0.40 | 11 | 5 |
| C | 4 | 20,565 | 0.841 | 0.526 | 0.88 / 0.33 / 0.86 / 0.01 / 0.78 | 369 | 2068 |

## 6. Per-fold diagnostics (descriptive; not used for any decision)

- **Model B, outer fold 3, F class:** 11 true F beats, 3,028 predicted F (275x), 1 true positives, precision 0.0003, recall 0.091; false positives by true class {'N': 2940, 'S': 24, 'V': 63, 'Q': 0}; top contributing patients (patient, F false positives): [('109', 1033), ('105', 957), ('201_202', 534), ('114', 461)]. On the validation partition the same model predicted F 3,951 times for 369 true F beats.
- **Model A, outer fold 3, F class:** 11 true F beats, 998 predicted F (91x), 2 true positives, precision 0.0020, recall 0.182; false positives by true class {'N': 978, 'S': 5, 'V': 13, 'Q': 0}; top contributing patients (patient, F false positives): [('201_202', 671), ('114', 213), ('105', 100), ('200', 9)]. On the validation partition the same model predicted F 1,323 times for 369 true F beats.
- **Model C, outer fold 3, F class:** 11 true F beats, 76 predicted F (7x), 2 true positives, precision 0.0263, recall 0.182; false positives by true class {'N': 66, 'S': 0, 'V': 8, 'Q': 0}; top contributing patients (patient, F false positives): [('105', 46), ('114', 16), ('200', 6), ('201_202', 5)]. On the validation partition the same model predicted F 302 times for 369 true F beats.
- Patient-level concentration, class F: 802 beats; largest contributors [('208', 372, '46.4%'), ('213', 362, '45.1%'), ('223', 14, '1.7%')].
- Patient-level concentration, class S: 2,775 beats; largest contributors [('232', 1380, '49.7%'), ('209', 383, '13.8%'), ('222', 209, '7.5%')].
- Patient-level concentration, class Q: 8,027 beats; largest contributors [('102', 2080, '25.9%'), ('107', 2074, '25.8%'), ('104', 2060, '25.7%')].
- Fold 3's test partition has 5 Q beats and 11 F beats; fold 1 has 13 F beats; fold 2's validation partition (used for model selection) had 5 Q and 11 F beats; per-fold F and Q numbers are therefore unstable and not interpreted. B fold 3 (selected at epoch 1 on validation, flagged earlier) over-predicts F on both its validation and test partitions.

## 7. Independent verification

`final_revision/tools/verify_independent.py` recomputed everything from the saved score files with separate implementations (confusion-matrix metrics, rank-based AUROC, step-wise average precision, calibration, bootstrap with the protocol RNG, McNemar exact test) and compared with the evaluation outputs: **43 checks, 1 flagged on first run, 0 unresolved** (runtime 610 s).

| Check | Result |
|---|---|
| A: pooled predictions = 109,305, every beat index exactly once (0..109304) | PASS |
| A: stored labels / subjects / records equal the source data at those beat indices | PASS |
| A: each fold's test subjects == locked fold k; subjects are disjoint across folds; 201/202 grouped as 201_202 | PASS |
| A: every record belongs to exactly one subject (48 records, 47 subjects) | PASS |
| A: true-label counts == data-audit counts and == frozen protocol (N 90,472 / S 2,775 / V 7,229 / F 802 / Q 8,027) | PASS |
| A: predicted-label counts | PASS |
| B: pooled predictions = 109,305, every beat index exactly once (0..109304) | PASS |
| B: stored labels / subjects / records equal the source data at those beat indices | PASS |
| B: each fold's test subjects == locked fold k; subjects are disjoint across folds; 201/202 grouped as 201_202 | PASS |
| B: every record belongs to exactly one subject (48 records, 47 subjects) | PASS |
| B: true-label counts == data-audit counts and == frozen protocol (N 90,472 / S 2,775 / V 7,229 / F 802 / Q 8,027) | PASS |
| B: predicted-label counts | PASS |
| C: pooled predictions = 109,305, every beat index exactly once (0..109304) | PASS |
| C: stored labels / subjects / records equal the source data at those beat indices | PASS |
| C: each fold's test subjects == locked fold k; subjects are disjoint across folds; 201/202 grouped as 201_202 | PASS |
| C: every record belongs to exactly one subject (48 records, 47 subjects) | PASS |
| C: true-label counts == data-audit counts and == frozen protocol (N 90,472 / S 2,775 / V 7,229 / F 802 / Q 8,027) | PASS |
| C: predicted-label counts | PASS |
| per-fold table (n, accuracy, macro-F1 over classes present, per-class sensitivity, per-class counts) == independent | PASS |
| [mit_internal] A: point metrics (accuracy, macro-F1, balanced acc., macro AUROC/AUPRC, ECE, signed gap, MCE, class-wise ECE, Brier, NLL, raw-score calibration) == independent | PASS |
| [mit_internal] A: 5x5 confusion matrix == independent | PASS |
| [mit_internal] A: row-normalised confusion matrix | PASS |
| [mit_internal] A: reliability bins (n, confidence, accuracy, signed gap) | PASS |
| [mit_internal] A: per-class sensitivity, specificity, PPV, F1, AUROC, AUPRC, beat and cluster support, primary-macro flags == independent | PASS |
| [mit_internal] B: point metrics (accuracy, macro-F1, balanced acc., macro AUROC/AUPRC, ECE, signed gap, MCE, class-wise ECE, Brier, NLL, raw-score calibration) == independent | PASS |
| [mit_internal] B: 5x5 confusion matrix == independent | PASS |
| [mit_internal] B: row-normalised confusion matrix | PASS |
| [mit_internal] B: reliability bins (n, confidence, accuracy, signed gap) | PASS |
| [mit_internal] B: per-class sensitivity, specificity, PPV, F1, AUROC, AUPRC, beat and cluster support, primary-macro flags == independent | PASS |
| [mit_internal] C: point metrics (accuracy, macro-F1, balanced acc., macro AUROC/AUPRC, ECE, signed gap, MCE, class-wise ECE, Brier, NLL, raw-score calibration) == independent | PASS |
| [mit_internal] C: 5x5 confusion matrix == independent | PASS |
| [mit_internal] C: row-normalised confusion matrix | PASS |
| [mit_internal] C: reliability bins (n, confidence, accuracy, signed gap) | PASS |
| [mit_internal] C: per-class sensitivity, specificity, PPV, F1, AUROC, AUPRC, beat and cluster support, primary-macro flags == independent | PASS |
| [mit_internal] A: 95% subject-cluster bootstrap CIs (accuracy, macro-F1, balanced acc., macro AUROC, ECE, signed gap, Brier) == independent bootstrap (B=2000) | PASS |
| [mit_internal] A: per-class sensitivity / PPV / F1 CIs and number of valid resamples == independent | PASS |
| [mit_internal] B: 95% subject-cluster bootstrap CIs (accuracy, macro-F1, balanced acc., macro AUROC, ECE, signed gap, Brier) == independent bootstrap (B=2000) | PASS |
| [mit_internal] B: per-class sensitivity / PPV / F1 CIs and number of valid resamples == independent | PASS |
| [mit_internal] C: 95% subject-cluster bootstrap CIs (accuracy, macro-F1, balanced acc., macro AUROC, ECE, signed gap, Brier) == independent bootstrap (B=2000) | PASS |
| [mit_internal] C: per-class sensitivity / PPV / F1 CIs and number of valid resamples == independent | PASS |
| [mit_internal] paired B - A: delta macro-F1 + paired CI, per-class delta-sensitivity CIs, McNemar discordant counts and exact p == independent | PASS |
| [mit_internal] paired B - C: delta macro-F1 + paired CI, per-class delta-sensitivity CIs, McNemar discordant counts and exact p == independent | **FLAGGED** |
| [mit_internal] paired C - A: delta macro-F1 + paired CI, per-class delta-sensitivity CIs, McNemar discordant counts and exact p == independent | PASS |

**Investigation of the flagged check ([mit_internal] paired B - C).** frozen evaluation outputs are written with pandas to_json (double_precision=10); the stored p-value 0.0047042146 differs from the exact 0.004704214593638442 by 4e-11 (> 1e-12). Pairs B-A and C-A passed because their p-values (1e-18, 1e-42) round to ~0. Components compared: delta macro-F1, paired CI (2), McNemar discordant counts (2, exact integers), exact p, per-class delta-sensitivity CIs (5 x 2). Maximum absolute difference over all components: 4.81e-11; discordant counts identical. Resolution: All components agree with the independent computation to <= 4.5e-11 (storage rounding). The original check tolerance was too tight for the stored precision. verification_mit.json is preserved unchanged. (record: `results/verification_mit/discrepancy_B_minus_C_resolution.json`; `verification_mit.json` is preserved unchanged.)

## 8. Warnings and limitations

- Rare classes: S (2,775 beats, 31 patients, 50% from record 232) and F (802 beats, 16 patients, 91.5% from records 208 and 213) have very wide intervals; Q (8,027 beats) comes from 9 patients of which only 4 are paced and 3 contribute ~77% of the beats (two paced records use lead V5). Class-level numbers must not be generalised.
- Bootstrap CIs reflect test-set (patient) sampling uncertainty of fixed models; they do not include training-seed variability (one seed per fold) or fold-model differences, and are unstable for classes concentrated in few patients.
- MIT-BIH is an *internal* patient-grouped evaluation, not a pristine hold-out: every record has been a test record in earlier analyses (disclosed in the protocol). Folds were assigned using per-patient label counts only.
- Model B is a PyTorch port of the Keras architecture; models differ in epoch budget (15/20/15) and architecture, so the comparison is a comparison of experimental configurations. The paired macro-F1 intervals for B-A and B-C include 0.
- Per-fold macro-F1 averages only classes present (>= 1 beat), unlike the pooled five-class macro; per-fold numbers carry no CIs and are not compared.
- ECE is dominated by class N and by fold-wise temperature scaling fitted on validation folds with very different class mixes (fold 2 validation: 11 F and 5 Q beats); pooled ECE is a mixture of five temperatures.
- Validation macro-F1 selection was noisy (B fold 3 selected epoch 1; fold-2 validation metric averaged four classes); these choices were fixed before scoring and were not changed.
- Model C inference used batch 4096 with shared-memory spill on an 8 GB GPU; results are deterministic given the checkpoints (validation logits reproduced bitwise earlier).
