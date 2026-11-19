# R2 Protocol Addendum A1: separated, staged scoring and evaluation

Status: **PREPARED, not yet exercised on any test or external data.** Created 2026-11-06 at the authors' instruction ("Option 2: separate MIT-BIH and INCART scoring completely").
This addendum does **not** amend `experimental_protocol_LOCKED.md`, `locked_protocol.json`, `folds_LOCKED.json`, the training artefacts or any manifest. It documents a deviation in *how* the frozen scoring/evaluation computations are executed.

## 1. Why this addendum exists

The frozen `07_score.py` scores the MIT-BIH test folds **and** INCART in one execution (lines 16, 23-24, 28) under one completion flag, and the frozen `08_evaluate.py` main section needs both (lines 112, 129, 133). It has no option to run one dataset only. The authors require INCART to stay a separately authorised external stage. The only way to do that without editing frozen files is to execute the same computations through separate drivers. `07_score.py` and `08_evaluate.py` are therefore **not executed**; they remain in place, hash-locked and unmodified.

## 2. What is replaced by what

| Frozen code | Replacement | How the computation is kept identical |
|---|---|---|
| `07_score.py`, MIT-BIH part (lines 17-22, 25-27) | `experiments/07a_score_mit.py` | Same expressions (checked by AST comparison): `rc.build`, `torch.load(selected checkpoint)`, `rc.partition(...)` test mask, `rc.logits(model, d['X'][te])`, same keys `logits, y, subject, record, beat_index, T`, same model/fold order |
| `07_score.py`, INCART part (lines 16, 23-24, 28) | `experiments/07b_score_incart.py` | Same expressions (AST-checked): `logits`, `logits_recentred`, `T`; labels file `y, patient, record` |
| `08_evaluate.py` main, MIT-BIH block (lines 114-127) | `experiments/08a_evaluate_mit.py` | **Executes the frozen statements themselves** (extracted by AST from the hash-locked file, layout asserted); all metric code is the frozen `evaluate()`, bootstrap B = 2000, seed 20261102 |
| `08_evaluate.py` main, INCART block (lines 129-139) | `experiments/08b_evaluate_incart.py` | Same: frozen statements executed verbatim; `cls=[0,1,2,3]` (Q excluded from the INCART macro-F1, protocol §6/D6) |
| (new, shared) | `experiments/a1common.py` | Verification, guards, staging helpers; no scientific computation |

Unchanged: protocol, folds, seeds, hyper-parameters, preprocessing, selected checkpoints and temperatures, the metric definitions, the bootstrap, the pre-specified comparisons (B-A, B-C; C-A exploratory), the primary model (B), the class sets (five classes MIT-BIH, N,S,V,F INCART), and the rule that no test or INCART quantity feeds any decision.

## 3. Stage order and authorisation gates (each needs explicit approval and its own flag)

1. `python -P -B experiments/pre_scoring_checks.py pre07` (read-only pre-checks), then **MIT-BIH scoring**: `python -P -B experiments/07a_score_mit.py --authorised-mit-test-scoring` -> `results/scores_mit/`
2. **MIT-BIH primary evaluation**: `python -P -B experiments/08a_evaluate_mit.py --authorised-mit-evaluation` -> `results/evaluation_mit/`
3. Optional descriptive diagnostics (approved, only after step 2): `python -P -B experiments/10_per_fold_diagnostics.py --approved --scores-dir results\scores_mit --eval-dir results\evaluation_mit`
4. **INCART scoring** (separate authorisation; refuses unless steps 1-2 are complete and unchanged): `07b_score_incart.py --authorised-incart-scoring` -> `results/scores_incart/`
5. **INCART evaluation** (separate authorisation): `08b_evaluate_incart.py --authorised-incart-evaluation` -> `results/evaluation_incart/`

Superseded and not to be used: `07_score.py`, `08_evaluate.py` (as executables), `run_after_training.sh`, the `pre08` stage of `pre_scoring_checks.py`, `09_class_driver_analysis.py` (never run; not part of the evaluation chain).

## 4. Guarantees and how each was verified (validation data only; see section 6)

* **MIT-BIH stage cannot read or access INCART data.** `07a`, `08a`, `10_per_fold_diagnostics.py` and `pre_scoring_checks.py` install a process-level audit hook that raises `PermissionError` on any open/list/scan of a path containing "incart" (cannot be removed once installed). The single exception is **read-only** access to the two INCART *driver source files* (`experiments/07b_score_incart.py`, `experiments/08b_evaluate_incart.py`), which the addendum check must hash to detect tampering; write-mode access, look-alike names elsewhere, data files, score/evaluation directories and the raw database remain blocked (tested). The scoped lock check (`a1.verify_lock(include_external=False)`) hashes the other 12 frozen files and never opens the INCART data file. `07a`/`08a` contain no INCART identifier outside docstrings. Verified: the guard blocks open, listdir and scandir of the real INCART paths; a recorder saw no "incart" path opened in the whole test process.
* **The INCART stage cannot influence MIT-BIH results.** `07b`/`08b` install a write guard that makes any write, rename or delete under the MIT-BIH score/evaluation directories (final or staging) raise `PermissionError`; they run only if the MIT-BIH stage is complete and every MIT-BIH output still matches the hashes in its `SCORED.json`/`EVALUATED.json`; MIT-BIH directories are re-hashed before the INCART flag is written and again by `08b`. Verified: four write attempts blocked; MIT-BIH directories byte-identical before and after both INCART stages.
* **Write-once, interruption-safe.** Outputs are written to `*_INPROGRESS` directories (marker `_STARTED.json`, exclusive-create files), fully re-read and verified, the completion flag is written **last**, then the directory is renamed. Existing staging or final directories make the script refuse. On any failure the staging directory and logs are preserved and the script stops. Verified: refusals (final exists, staging exists), injected mid-run failure (7 of 15 files preserved, no flag, no final directory, re-run refuses).
* **Linked to the protocol.** Before touching data each driver verifies: the lock (scoped), this addendum's hashes (all new scripts, the four frozen scripts), every run file against `manifest_post_training.json` (including the 15 selected checkpoints, before loading), and the read-only backup. Flags record the lock hash, addendum hash, manifest hash, script hash, per-checkpoint SHA-256 and per-output SHA-256.
* **MIT-BIH coverage.** Before the flag is written `07a` checks that the five test folds cover all 109,305 beats exactly once, per-fold sizes equal the locked partition, labels/subjects equal the source data, logits are finite, and T equals the selected temperature.
* **Sleep.** Each driver holds a system-required request for its lifetime (no setting is changed; released on exit). The lid-close action is not covered.

## 5. Disclosures

* The frozen lock verification (used in all earlier steps, including training and the post-training audit) hashes the INCART data file to check integrity; no content was parsed. The MIT-BIH-phase scripts above do not read it. `07b`/`08b` do hash it, as part of the full lock check, because they are the INCART stage.
* `rc.load_mit()` (used by the frozen training code and by `07a`) loads the whole MIT-BIH array, test partitions included; partitions are selected by mask exactly as in the frozen scripts.
* `10_per_fold_diagnostics.py` and `pre_scoring_checks.py` were edited after their first hashing to use the scoped lock check and the access guard; their earlier hashes in `results/audits/pre_scoring_preparation.json` are superseded by the hashes in `protocol_addendum_A1.json`.
* The interpretation limits accepted in the pre-evaluation review (rare-class intervals, McNemar, per-fold numbers, ECE, fold-2 four-class validation metric, B fold 3 epoch 1) apply unchanged, because the frozen metric code is what runs.
* The manuscript must state that scoring and evaluation were executed through staged drivers (07a/08a, then 07b/08b) that run the frozen computations, instead of the combined scripts.

## 6. Verification performed before approval (validation data only; INCART never opened)

* MIT side (31 checks): 07a in validation mode on the five validation partitions reproduces the training-time validation logits **bitwise** (max abs difference 0); 08a on those pseudo-pooled scores matches scikit-learn and an independent bootstrap (point metrics, confusion matrices, per-class precision/recall/F1/support, paired CIs, McNemar counts, per-fold table) and is **byte-identical** to calling the frozen `evaluate()` directly; refusal and failure paths behave as specified.
* INCART side (24 checks): 07b/08b were run on a stand-in set drawn from MIT-BIH validation beats (the real INCART files were blocked at audit-hook level): outputs equal independent computations, the INCART block (Q excluded from the macro) is byte-identical to the frozen `evaluate()` on independently built ensembles, MIT-BIH directories are untouched, all guards fire.
* Defects found and fixed before hashing: (a) verification loops left `.npz` files open so Windows refused the final directory rename; (b) log-file names collided within one second; (c) the layout assertion for the frozen INCART block (line 129 holds two statements); (d) the MIT-stage access guard blocked the addendum check from hashing the two INCART driver source files, so a narrow read-only exception for exactly those two files was added (and the addendum files, created minutes earlier and never used, were regenerated). All re-tested after the fix.

Hashes of the new files, the four frozen scripts and this document are in `protocol_addendum_A1.json` (itself hashed in `protocol_addendum_A1.sha256`). The drivers refuse to run if any of them differs.
