# Reproducibility status

Assembled: 2026-11-10. Overall: **ready for review by the authors; NOT yet publicly deposited; two manual decisions outstanding (licence, deposit).**
Legend: PRESENT = included and hash-listed; PARTIAL = included with stated limits; RESTRICTED = not redistributed (instructions given); ABSENT = does not exist in the project; MANUAL = needs an action by the authors.

| # | requested material | status | where / evidence | note |
|---|---|---|---|---|
| 1 | Frozen protocol and its hash | PRESENT | `protocol/locked_protocol.json`, `locked_protocol.sha256`; `verify_archive.py` confirms the hash (ed0717f2...61478c) | addendum A1 hash also present |
| 2 | Exact patient/record fold assignments | PRESENT | `fold_assignments/folds_LOCKED.json` | records 201 and 202 form one patient |
| 3 | Preprocessing scripts | PRESENT | `preprocessing/04_extract_beats.py` (+ audits) | need the PhysioNet downloads (RESTRICTED, item 16) |
| 4 | Training scripts | PRESENT | `training/` | ECGTransForm code included under its MIT licence |
| 5 | Scoring scripts | PRESENT | `scoring/` | guarded by the protocol lock |
| 6 | Calibration scripts | PARTIAL | `training/r2common.py` (`fit_T`), `bootstrap/08_evaluate.py` (`calib`) | no standalone calibration script exists; calibration is part of training/evaluation code |
| 7 | Bootstrap scripts | PRESENT | `bootstrap/08_evaluate.py` | patient-cluster, 2,000 resamples, seed 20261102 |
| 8 | Software environment and package versions | PARTIAL | `environment/` | no lockfile or container; documented, unreconciled discrepancies (PyWavelets 1.8.0 module vs 1.10.0 pip metadata; NumPy from user site-packages) |
| 9 | MIT-BIH out-of-fold predictions | PRESENT | `predictions/mitbih/` | per-beat logits, 15 files |
| 10 | INCART predictions | PRESENT | `predictions/incart/` | per-beat logits, 15 files + labels |
| 11 | Result manifests | PRESENT (2 redacted) | `manifests/` | `manifest_pre_resume.json` and `manifest_post_training.json` differ from the originals only by one redacted user path; both hashes are listed in `REDACTION_NOTE.md` |
| 12 | Verification outputs | PRESENT | `verification/` | 43 MIT-BIH and 51 INCART independent checks recorded; stage audits are PASS WITH WARNINGS |
| 13 | Model-checkpoint hashes | PRESENT | `checkpoints/CHECKPOINT_HASHES.csv` | 45/45 recomputed hashes equal the locked-manifest hashes |
| 14 | Model checkpoints | PRESENT | `checkpoints/` (all 45, about 80 MB) | MANUAL: licence for the trained weights not yet chosen |
| 15 | Scripts for every revised figure and table | PARTIAL | `figure_table_generation/` | Figures 1-5 and Tables 2, 4, 5, 7-10 have generators; Tables 1, 3, 6 and 11 are descriptive text tables written by hand from the protocol, code and audits (no generator). Figure 5 needs the processed beat array (item 16) |
| 16 | Processed beat arrays / raw ECG | RESTRICTED (not included) | `data_access/DATA_ACCESS.md`, hashes in `manifests/processed_data_hashes.json` | open PhysioNet databases; not redistributed to avoid unnecessary data copies; regenerate with `04_extract_beats.py` |
| 17 | Persistent public identifier (DOI/URL) | ABSENT / MANUAL | - | create after deposit; add to the manuscript Data availability statement and the response letter |
| 18 | Licence for author-written code | ABSENT / MANUAL | - | choose (for example MIT or Apache-2.0) before release |
| 19 | Local path constants | KNOWN ISSUE | listed in README | frozen scripts/logs contain `D:\explainable_ai\...` constants; left unmodified to keep hashes valid; no user name or credential present |
| 20 | Legacy 300-sample Keras demonstration | EXCLUDED | project folder `clinical_ai` | not part of the evaluated study; no result in the manuscript depends on it |
| 21 | Container / one-command reproduction | ABSENT | - | not available; steps are documented in the README |

## Verification performed at assembly

* `verify_archive.py`: every file in `FILE_MANIFEST.csv` re-hashed; protocol lock hash; frozen-file hashes from the locked manifest; 45 checkpoint hashes (results recorded in `ASSEMBLY_VERIFICATION.txt`).
* Numeric tables regenerated from the archived evaluation outputs and compared with the manuscript tables (`figure_table_generation/tables_generated/`).
* Figure 5 re-run from the locked Model B fold-0 checkpoint: output image and provenance JSON are byte-identical to the previous build.

## Not verified

* Full retraining and re-scoring were not repeated for the archive (frozen experiments; no new experimental cycle).
* The archive scripts were not executed on a clean machine other than the original workstation.
