# Public reproducibility archive: patient-grouped ECG beat classification (MIT-BIH) with external evaluation on INCART

**Status: release candidate; NOT yet published.** No DOI exists yet; do not cite as publicly available until a release has been published.

Authors: Sreyas Sanil Kumar; Kishore N.; Amritha Lakshmi T. M.; Rahul S. G.; Venkatasubramanian K. B..

Licence: MIT for original code and documentation (`LICENSE`); third-party materials and data terms in `THIRD_PARTY_NOTICES.md` and `data_access/DATA_ACCESS.md`.

Accompanies the manuscript *"Patient-Grouped Deep Learning for ECG Beat Classification: Explainability Components and External Evaluation on INCART"*.
The archive contains only artifacts that already existed in the project: the frozen protocol, fold assignments, scripts, out-of-fold and external predictions, result manifests,
verification outputs and model checkpoints, plus the scripts that generate the manuscript figures and numeric tables. Nothing was retrained, re-scored or re-evaluated for the archive,
and no result file was modified.

## Layout

| folder | content |
|---|---|
| `protocol/` | locked protocol (md + json), its SHA-256, addendum A1 (separation of MIT-BIH and INCART stages) |
| `fold_assignments/` | locked patient/record-to-fold assignment, design script, per-fold class support, per-patient class counts |
| `preprocessing/` | audit scripts and `04_extract_beats.py` (db5 level-9 denoising, per-record z-score, R-99..R+200 window, resampling to 186 samples), audit outputs, logs |
| `training/` | shared code (`r2common.py`: models A/B, SMOTE, training loop, temperature fit), driver, lock script, ECGTransForm model code (MIT licence), per-run configs / histories / selections / validation logits, logs, documentation of one interrupted unused attempt |
| `scoring/` | scoring scripts and logs for MIT-BIH out-of-fold and INCART scoring |
| `calibration/` | reliability-bin tables and a pointer to the calibration code (`training/r2common.py: fit_T`, `bootstrap/08_evaluate.py: calib`) |
| `bootstrap/` | evaluation scripts (metrics, calibration, patient-cluster bootstrap, paired comparisons) and all evaluation outputs (CSV/JSON) |
| `predictions/mitbih/` | per-beat logits of the pooled out-of-fold MIT-BIH predictions (`mit_<model>_fold<k>.npz`) |
| `predictions/incart/` | per-beat logits of each fold model on INCART, plus `incart_labels.npz` |
| `manifests/` | result manifests (two redacted copies; see `manifests/REDACTION_NOTE.md`), final results manifest, hashes of the (not redistributed) processed beat arrays |
| `verification/` | independent recomputation script and outputs, stage audits, final results report/summary, plots |
| `checkpoints/` | all 45 PyTorch checkpoints (3 learning rates x 15 fold models) and `CHECKPOINT_HASHES.csv` (locked-manifest hash vs recomputed hash) |
| `figure_table_generation/` | scripts for Figures 1-5 and Tables 2, 4, 5, 7-10, regenerated tables, final figure images |
| `environment/` | software/hardware environment at training time and relevant package versions |
| `data_access/` | how to obtain MIT-BIH and INCART from PhysioNet, citations, licence note |
| `FILE_MANIFEST.csv` | relative path, size, SHA-256 and description of every file |
| `REPRODUCIBILITY_STATUS.md` | what is present / absent / restricted / needs manual action |
| `ARCHIVE_RELEASE_CHECKLIST.md` | steps to publish the archive |
| `verify_archive.py`, `reconstruct_original_layout.py`, `PATH_MAP.csv` | integrity check and layout reconstruction |

## Reproduction steps

Requirements: Python 3.13, the packages in `environment/requirements_relevant.txt`. Steps are ordered from cheap to expensive.

1. **Integrity.** `python verify_archive.py` checks every file against `FILE_MANIFEST.csv`, the protocol lock, the frozen-file hashes in the locked manifest and all 45 checkpoint hashes.
2. **Tables.** `python figure_table_generation/generate_tables.py --out tables_generated` regenerates Tables 2, 4, 5, 7, 8, 9 and 10 from `bootstrap/` (the copy in `figure_table_generation/tables_generated/` is the archived output).
   `python figure_table_generation/check_manuscript_tables.py --docx <manuscript.docx> --tables tables_generated` compares a manuscript with them and re-derives the in-text failure-mode numbers from the locked confusion matrices and predictions.
3. **Independent metric check.** `verification/verify_independent.py` recomputes every reported metric from `predictions/` with its own implementation (it expects the original layout, see below; it writes into `results/verification_*`, so run it in a fresh reconstructed tree after removing the existing verification folder there).
4. **Figures.** Reconstruct the original layout (below), set `R2_ROOT` to it, then run `figure_table_generation/architecture/make_architecture_figures.py` (Figures 2-4; needs `ECGTF_MODELS_PY` pointing to `training/ecgtransform/models.py` and the ECGTransForm folder on `sys.path` as in `r2common.py`),
   `figure_table_generation/make_figure1_beat_extraction.py` (Figure 1; set `MITBIH_DIR`) and `figure_table_generation/ig_figure/make_figure5_ig.py` (Figure 5; needs the processed beat array, step 6).
5. **Re-scoring from checkpoints.** `scoring/07a_score_mit.py` and `07b_score_incart.py` (guarded by the protocol lock) regenerate `predictions/` from `checkpoints/`; GPU recommended.
6. **Processed beat arrays.** Download MIT-BIH and INCART (`data_access/DATA_ACCESS.md`) and run `preprocessing/04_extract_beats.py`; compare with the hashes in `manifests/processed_data_hashes.json`.
7. **Retraining.** `training/06_train.py` (one model x fold; about 1-2 min per learning rate for A, 6 min for B and 2 h for C per fold on the recorded laptop GPU). Fixed seeds and deterministic cuDNN are used, but bit-identical retraining on other hardware is not guaranteed.

### Reconstructing the original layout and the path constants

The frozen scripts expect the project's original folder tree and contain local path constants. They were copied byte-identically so that the hashes recorded in the locked manifest still verify.
`python reconstruct_original_layout.py <target_dir>` copies the archive into the original layout (`experiments/`, `results/`, `data/`, ...). Before running a script from the reconstructed tree, edit these constants:

* `experiments/r2common.py` line 8: `ECGTF` (folder with the upstream ECGTransForm code; use `training/ecgtransform/` content);
* `experiments/01_audit_mitbih.py`, `03_audit_incart.py`, `03b_mitbih_alignment_by_class.py`, `04_extract_beats.py`: `MIT` (PhysioNet MIT-BIH folder);
* `final_revision/tools/verify_independent.py`: `R2` (root of the reconstructed tree);
* INCART is expected in `<root>/data/incartdb/` (PhysioNet download including `files-patients-diagnoses.txt`).

Editing a frozen script changes its hash; use the unmodified archive copies for hash verification.
Logs and manifests contain the original local paths (`D:\explainable_ai\...`); no user name or credential is present (the only user-profile path was redacted, see `manifests/REDACTION_NOTE.md`).

## Scope and caveats

* Reported results correspond to one training seed per fold; confidence intervals cover test-set sampling (patient-cluster bootstrap) only.
* The legacy 300-sample Keras reporting demonstration is **not** part of the evaluated study and is not included.
* ECGTransForm code is third-party (MIT licence, (c) 2023 Emadeldeen Eldele); its licence is included.
* Original code and documentation: MIT licence. Trained checkpoints and derived predictions are released under the same MIT licence for the authors' own contribution (see THIRD_PARTY_NOTICES.md for the underlying data terms).

## Citation

See `CITATION.cff`. A DOI will be added only after a Zenodo release exists.
