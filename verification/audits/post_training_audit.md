# Post-training audit: ECG arrhythmia revision R2

- Audit date: 2026-11-06 (all times IST). Scope: audit only. No test scoring, no MIT-BIH test predictions, no INCART evaluation, no post-hoc analysis.
- Protocol lock: `ed0717f26f2d7f86d5a49c69b03de44588d35112ab8c9177186b4c4e5b61478c`. Training chain finished `RESUME FINISHED: 15/15` at 2026-11-06T00:50:48+05:30.
- Companion manifest: `results/manifests/manifest_post_training.json` (SHA-256 of every output file; pre-resume manifest preserved unchanged).

## 1. Overall verdict: **PASS WITH WARNINGS**

All 492 automated checks passed (0 failed). Nothing needs repair. The warnings (section 8) concern the *statistical reliability* of the validation signal used for selection (very small F and Q validation counts in two folds; noisy epoch-1 selection for B fold 3) and operational/documentation points. None of them is a protocol violation, and none may be "fixed" without breaking the frozen protocol; they are flagged for scientific review.

## 2. Completion table (15 runs)

| Run | Seed | Status | Selected lr | Epoch | Val macro-F1 | Val classes | T | Ckpts | Val logits | Compute (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| A fold 0 | 1000 | COMPLETE | 0.0001 | 1 | 0.5487 | NSVFQ | 1.005 | 3 | 1 | 1.2 |
| A fold 1 | 1001 | COMPLETE | 0.001 | 4 | 0.5554 | NSVFQ | 3.609 | 3 | 1 | 1.2 |
| A fold 2 | 1002 | COMPLETE | 0.001 | 3 | 0.4259 | NSVF | 1.848 | 3 | 1 | 1.3 |
| A fold 3 | 1003 | COMPLETE | 0.001 | 13 | 0.4730 | NSVFQ | 5.150 | 3 | 1 | 2.7 |
| A fold 4 | 1004 | COMPLETE | 0.001 | 3 | 0.4419 | NSVFQ | 1.828 | 3 | 1 | 1.3 |
| B fold 0 | 1000 | COMPLETE | 0.0005 | 11 | 0.5605 | NSVFQ | 2.307 | 3 | 1 | 6.1 |
| B fold 1 | 1001 | COMPLETE | 0.001 | 20 | 0.4661 | NSVFQ | 3.201 | 3 | 1 | 5.9 |
| B fold 2 | 1002 | COMPLETE | 0.001 | 15 | 0.3795 | NSVF | 3.009 | 3 | 1 | 6.2 |
| B fold 3 | 1003 | COMPLETE | 0.0005 | 1 | 0.5084 | NSVFQ | 1.815 | 3 | 1 | 6.3 |
| B fold 4 | 1004 | COMPLETE | 0.001 | 4 | 0.5001 | NSVFQ | 2.256 | 3 | 1 | 6.6 |
| C fold 0 | 1000 | COMPLETE | 0.0005 | 14 | 0.5232 | NSVFQ | 3.231 | 3 | 1 | 119.6 |
| C fold 1 | 1001 | COMPLETE | 0.001 | 8 | 0.5624 | NSVFQ | 4.088 | 3 | 1 | 129.9 |
| C fold 2 | 1002 | COMPLETE | 0.0005 | 9 | 0.3727 | NSVF | 2.287 | 3 | 1 | 112.4 |
| C fold 3 | 1003 | COMPLETE | 0.0005 | 11 | 0.4744 | NSVFQ | 2.605 | 3 | 1 | 114.0* |
| C fold 4 | 1004 | COMPLETE | 0.001 | 12 | 0.5153 | NSVFQ | 2.311 | 3 | 1 | 118.0 |

Totals: 15 runs, 45 checkpoints, 15 `selected.json`, 15 `val_logits.npz`, 45 history files, 15 configs, 15 train logs. All present, readable and verified.
\* C fold 3 wall time recorded as 25,355 s for lr 1e-3; compute is estimated at 2,300 s (section 8, item C). Compute minutes = sum of the three per-lr times from the training logs.
Val classes = classes with >=10 validation beats (the locked rule); "NSVF" means Q was excluded. Val macro-F1 is on the validation partition only.

## 3. Checkpoint and validation-logit integrity

For every run (all 15):

- `selected.json`, `config.json`, `val_logits.npz`, 3 checkpoints and 3 histories exist; checkpoints load with `torch.load(weights_only=True)`, keys and shapes match the model, all values finite.
- Each history has the full epoch count (A 15, B 20, C 15), contiguous epochs, finite losses: no early stopping or truncated run. The recorded `val_macro_f1` equals the mean of the recorded per-class F1 over the validation classes.
- `selected.json` grid, selected lr and epoch equal the argmax of validation macro-F1 over (lr, epoch) recomputed from the histories (first maximum, as in the locked code).
- Saved validation logits: shape (n_val, 5), finite, labels/subjects/records equal the validation partition; macro-F1 recomputed from them equals the recorded best F1 (< 1e-6); the temperature re-fitted from them equals the recorded T (< 1e-3).
- Independent check: each selected checkpoint was re-run on CPU on the validation partition only. Max |logit difference| vs the saved GPU logits and F1 difference:

| Run | max abs logit diff | argmax agreement | F1 abs diff |
|---|---|---|---|
| A fold 0 | 0.0016 | 1.00000 | 0.00e+00 |
| A fold 1 | 0.0056 | 0.99996 | 1.02e-05 |
| A fold 2 | 0.0059 | 0.99991 | 1.58e-05 |
| A fold 3 | 0.0060 | 1.00000 | 0.00e+00 |
| A fold 4 | 0.0049 | 0.99996 | 7.10e-05 |
| B fold 0 | 0.0119 | 1.00000 | 0.00e+00 |
| B fold 1 | 0.0151 | 0.99987 | 1.18e-04 |
| B fold 2 | 0.0102 | 1.00000 | 0.00e+00 |
| B fold 3 | 0.0046 | 1.00000 | 0.00e+00 |
| B fold 4 | 0.0066 | 1.00000 | 0.00e+00 |
| C fold 0 | 0.0123 | 0.99986 | 4.35e-05 |
| C fold 1 | 0.0075 | 0.99991 | 6.62e-05 |
| C fold 2 | 0.0096 | 0.99991 | 7.37e-05 |
| C fold 3 | 0.0064 | 0.99990 | 3.60e-06 |
| C fold 4 | 0.0053 | 1.00000 | 0.00e+00 |

The differences are CPU/GPU floating-point effects (largest F1 difference ~1.2e-4); the selected checkpoint files are the ones the recorded selection refers to.
- C fold 2 reproducibility: the re-run lr 1e-4 history is **identical** to the preserved interrupted attempt and its checkpoint is **byte-identical**: True. Training is deterministic in this environment.

Automated check groups (passed/total):

| Check | Passed / total |
|---|---|
| verify_lock (13 frozen files + lock self-hash) | 1/1 |
| locked_protocol.folds == folds_LOCKED.folds | 1/1 |
| 201 and 202 grouped as one unit | 1/1 |
| 47 units, 5 folds, disjoint | 1/1 |
| locked fold seeds 1000..1004 | 1/1 |
| lock config == r2common.CONFIG (as loaded now) | 1/1 |
| all 9 expected files present | 15/15 |
| seed == 1000+fold | 15/15 |
| config.json lock hash == current lock | 15/15 |
| config.json CONFIG == locked CONFIG | 15/15 |
| test_partition_used == "no"; SMOTE on training only | 15/15 |
| post-SMOTE train counts == 20000 x 5 | 15/15 |
| train/val subjects == locked-fold definition | 15/15 |
| train, val, test subject sets pairwise disjoint | 15/15 |
| train subjects exclude test fold | 15/15 |
| val class counts == config.json | 15/15 |
| val_classes_used == classes with >=10 val beats | 15/15 |
| history | 135/135 |
| selected.json grid == recomputed (best F1, first-max epoch) per lr | 15/15 |
| selected lr/epoch == argmax over (lr, epoch) of VALIDATION macro-F1 (ties -> first) | 15/15 |
| selected checkpoint filename matches selected lr | 15/15 |
| checkpoint | 45/45 |
| val_logits.npz loads; shape (n_val,5); finite | 15/15 |
| val_logits labels/subject/record == data[val partition] | 15/15 |
| macro-F1 recomputed from saved selected-checkpoint val logits == recorded best val macro-F1 | 15/15 |
| temperature re-fit from saved val logits == recorded T | 15/15 |
| train log present, 3 per-lr lines + SELECTED line | 15/15 |
| new | 2/2 |
| interrupted attempt preserved | 1/1 |
| pre-resume manifest: all 87 recorded hashes still match current files | 1/1 |
| absent | 7/7 |
| results/evaluation empty | 1/1 |
| 06_train.py | 1/1 |
| diag_memory.py | 1/1 |
| train_one() | 1/1 |
| resume script has no reference to 07/08/09/run_after_training | 1/1 |
| resume log | 1/1 |
| no process of 06/07/08/09 or chain scripts alive | 1/1 |
| driver version unchanged vs pre-resume manifest | 1/1 |
| all package versions == lock environment block | 1/1 |
| no package directory changed in either site-packages since 2026-11-01 | 1/1 |

## 4. Protocol-lock verification

- `verify_lock.py`: **OK** (`ed0717f26f2d7f86...`, 13 frozen files, lock self-hash matches `locked_protocol.sha256`).
- `locked_protocol.folds` equals `folds_LOCKED.json`: 5 folds, 47 units, disjoint; records 201 and 202 are one unit (`201+202`, fold 3).
- Fold seeds 1000-1004 and bootstrap seed 20261102 as locked; every run's `seed` = 1000 + fold.
- Every run's embedded configuration equals the locked CONFIG (batch 128, Adam, wd 1e-4, lr grid, epochs, SMOTE target 20,000/k=5, unweighted CE, checkpoint rule, temperature rule); every run's `lock_sha256` equals the current lock; post-SMOTE class counts are 20,000 x 5 in every run.
- Train/val subjects of every run equal the locked-fold definition (test = fold k, val = fold (k+1) mod 5, train = rest); train, val and test subject sets are pairwise disjoint.
- Frozen scripts (`04`, `06`, `07`, `08`, `r2common`, `verify_lock`) and the three ECGTransForm files still match their hashes. Nothing about preprocessing, SMOTE, architecture or training configuration changed during the resumed runs.

## 5. Manifest verification

- `manifest_pre_resume.json` preserved unchanged: SHA-256 `6d564a8cb46fb89c99ad76e2781acd457bb7f63353346ab75311a62d0ca3f563`, mtime 2026-11-05T12:22:30; all 87 file hashes it records (9 earlier runs, interrupted attempt) still match the current files.
- `manifest_post_training.json` (new, exclusive-create) records for all 15 runs: seed, selection, val class counts, every file's SHA-256 (45 checkpoints, 15 configs, 15 selection files, 15 val-logit files, 45 histories), CPU-recompute results, train-log hash; plus the interrupted attempt, logs, scripts, splits, protocol files, environment, events and all audit checks.

## 6. Environment and reproducibility

- Interpreter `C:\Python313\python.exe`; Python 3.13.7; NumPy 2.5.2 (user site-packages); torch 2.6.0+cu124 / CUDA 12.4 / cuDNN 90100; scikit-learn 1.7.2; imbalanced-learn 0.14.2; SciPy 1.16.3; wfdb 4.3.1; GPU NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, driver 617.42.
- All package versions equal the lock's environment block; driver unchanged since the pre-resume manifest; no package directory changed in either site-packages since 2026-11-01 (the last package change before the lock was 2026-10-14). Runs 0-2 (2026-11-02/03) and the resumed runs (2026-11-05/06) therefore ran in the same software environment.
- **Discrepancies (documented, not reconciled):** (a) PyWavelets module version 1.8.0 (= lock) vs pip metadata 1.10.0; preprocessing outputs are hash-frozen so training is unaffected. (b) NumPy 2.5.2 is loaded from the user site and shadows the system numpy 2.4.4; running with `-I`/`PYTHONNOUSERSITE` would load 2.4.4. (c) Configs record no package versions and the lock omits the driver version.

## 7. Preservation checks

- Interrupted C fold 2 attempt: `results/runs_interrupted/C_fold2_attempt1/` intact (6 files, hashes in manifest); not used for any result.
- Outside `R2/`, only two files under `D:\explainable_ai` have an mtime after 2026-10-13: `ECGTransForm/.git/objects/pack/pack-ad33...pack` (11-02 13:17) and `revision/reviewer_audit.md` (11-02 13:23). Both predate the R2 workspace (13:49) and this session's work. No manuscript file and no October experiment output has a modification time after 2026-10-12. *Limitation:* there is no earlier hash baseline for those files, so this is an mtime-based check.
- Inside `R2/` after the lock: only expected additions (runs, logs, manifests, diagnostics, helper scripts). No frozen file changed; no stray bytecode was created by this audit.

### Evaluation status

- No `results/scores/`, no `SCORED.json`, no `07_score`/`08_evaluate`/`09_posthoc` logs, `results/evaluation/` empty, no process of those scripts alive; the resume log shows only training runs (all `rc=0`) and ends with "Test scoring NOT run".
- Code inspection: `06_train.py` assigns the test mask but never uses it; `diag_memory.py` discards it; gradient steps use only the SMOTE-balanced training partition; validation data enter only the `no_grad` logits path; the resume script contains no reference to scripts 07/08/09. `run_after_training.sh` was never running during the resumed phase.
- Selection traceability: lr, epoch and checkpoint were chosen only from validation macro-F1 (recomputed from histories and from saved logits); T was fitted on the same validation logits. The test partition was never indexed, so no test result can have influenced any choice.

## 8. Anomalies, warnings and unresolved issues

**A. Model B fold 3 selected epoch 1 (flag for scientific review; selection unchanged)**

- Selected: lr 5e-4, epoch 1, validation macro-F1 0.5084 (T = 1.815). Verified: full 20-epoch histories for all three lrs, no early stopping or truncation; the selected checkpoint reproduces the recorded F1 exactly (saved logits and CPU recompute). It is the true validation maximum, not an implementation error.
- Other epochs: lr 5e-4 epochs 2-20 score 0.34-0.45; lr 1e-3 peaks at 0.478 (epoch 10); lr 1e-4 at 0.454 (epoch 18). The validation curves are very noisy (range ~0.30-0.51).
- Training loss falls monotonically (lr 5e-4: 0.47 to 0.017) while validation NLL is lowest at epoch 1 (0.99) and rises afterwards: strong subject-level overfitting. The epoch-1 checkpoint is barely trained; its selection is likely a winner's-curse effect of maximising a noisy validation metric over 60 (lr, epoch) candidates.
- Per-class validation F1 of the selected checkpoint (N,S,V,F,Q): 0.831, 0.163, 0.691, 0.074, 0.783; it predicts F 3,951 times for 369 true F beats (very low F precision).
- Same pattern in a milder form: A fold 0 also selected epoch 1. Selected epochs elsewhere span 3-20.

**B. Fold 2 validation set: missing/rare classes (correction to the brief and metric behaviour)**

- The fold-2 validation partition (fold 3 subjects) is not Q-free: counts N/S/V/F/Q = [19599, 323, 1626, 11, 5]. Q has **5** beats (below the locked threshold of 10) and F has only **11** (just above it). Fold 0's validation set has F = 13.
- Metric (`r2common.train_one`): `f1_score(y, pred, labels=<classes with >=10 val beats>, average='macro')`. Classes below the threshold are **excluded** from the average (not scored as zero). The per-class list that is also logged covers all five classes, with Q recorded as 0.0 for fold 2, but Q is not in the mean (verified: recorded macro-F1 equals the mean over N,S,V,F).
- Effect: averaging all five classes would scale fold-2 scores by 4/5: A 0.426 vs 0.341, B 0.380 vs 0.304, C 0.373 vs 0.298. Fold-2 validation F1 values are therefore averages over 4 classes and are **not comparable** with the other folds' 5-class values.
- The F class in fold 2 (11 beats) is included, so a quarter of the selection metric rests on 11 beats, with F1 of 0.002 / 0.007 / 0.000 (A/B/C). Selection in fold 2 (and F in fold 0, 13 beats) is correspondingly noisy. Models predict Q 1,051 / 381 / 580 times in the fold-2 validation set although only 5 are true Q: these false positives lower the recall of other classes (counted) but cannot lower a Q score (not counted).
- Temperature calibration: `fit_T` minimises NLL over all validation samples with five-class logits; the 5 Q beats contribute normally and nothing breaks. Re-fitted T equals the recorded T (A 1.848, B 3.009, C 2.287). Implication: T for fold 2 is estimated on a distribution with almost no F or Q beats, so its transfer to the fold-2 test set (which has F and Q) is uncertain.
- Nothing was changed; the rule is part of the frozen protocol. This should be stated in the manuscript and the fold-2 caveat considered in analysis planning.

**C. Model C fold 3: laptop suspension and inflated wall time**

- Checkpoints for all three lrs exist and load; selected lr 5e-4, epoch 11, validation macro-F1 0.4744 (lr 1e-4: 0.4545, lr 1e-3: 0.4726); T = 2.605.
- The training log records 25,355.4 s for lr 1e-3. Windows logged sleep at 15:55:08 (clock jump +14,510.5 s on resume at 19:57:05) and hibernate-from-sleep at 19:57:09 (clock jump +8,544.8 s on resume at 22:19:51), because the keep-awake wrapper had been terminated under memory pressure. Subtracting both jumps leaves about **2,300 s** of compute, matching the 2,268-2,270 s of the other lrs. The elapsed time must not be treated as compute time.
- The process and its GPU context survived suspend and hibernate; there were no GPU-reset events (id 153) in this phase, no error, and the outputs verify (history complete, logits and CPU recompute match the recorded F1).

**D. Other warnings**

1. Environment fragility: NumPy 2.5.2 comes from the user site and shadows the system 2.4.4; PyWavelets metadata vs module mismatch; no package versions in per-run configs (section 6).
2. Operational record gaps (no effect on results): `resume_monitor.log` has entries only at 12:23 (a `tail -F` helper locked the file; later the monitor was reaped), no entry after 15:54; `resume_wrapper.log` has no "released" line because the wrapper was killed; `run_training.log` contains the misleading line "FOLD 2 COMPLETE" after the failed C fold 2 (README in the interrupted folder explains it).
3. Unfrozen files outside the lock: `run_training_resume.sh` (one-line lock-check fix, diff recorded in the session), `run_resume_with_awake.ps1`, `monitor_resume.ps1`, `make_manifest_pre_resume.py`, `diag_memory.py`, and `09_class_driver_analysis.py` (created after the lock, **never run**, post-hoc analysis). None affects training. `09` should be reviewed against the protocol before it is ever run.
4. Minor timing noise: A fold 3 lr 1e-3 took 98.5 s (vs ~25 s) and B fold 4 lr 5e-4 142.8 s (vs ~125 s), while other work was running. No effect on results.
5. Model C validation inference (batch 4096) uses ~13.9 GB CUDA memory on the 8 GB GPU (system-memory spill). Unchanged from the folds completed before; the same code path (batch 4096) will be used by `07_score.py`.

## 9. Readiness for the one-time evaluation

**Recommendation: ready, subject to your decisions below.** The 15 locked fold models are complete, internally consistent, reproducible from their files, and untouched by any test or INCART data. Items to settle before approving `07_score.py` -> `08_evaluate.py`:
1. Acknowledge warnings A and B (noisy selection signal; fold-2 4-class validation metric; B fold 3 epoch 1). They cannot be changed under the frozen protocol; decide how they will be reported.
2. `07_score.py` writes `SCORED.json` only at the very end. A crash mid-run leaves partial logit files and no flag, so a re-run would overwrite them. I recommend keeping the laptop on AC with sleep/hibernate prevented, closing other apps (Model C inference spills to shared memory), and running it in the foreground once approved.
3. Decide whether `09_class_driver_analysis.py` (never run) belongs to the pre-specified analysis; keep it out of the evaluation chain until reviewed.
4. Optionally keep a read-only copy of `results/runs/` and the two manifests before scoring (a write outside the working set; not done).

No evaluation has been started. Waiting for your separate approval.
