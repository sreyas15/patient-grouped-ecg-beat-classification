# Data access

No raw ECG recordings are included. Download the databases from PhysioNet and cite the original publications:

1. **MIT-BIH Arrhythmia Database** (v1.0.0), https://physionet.org/content/mitdb/1.0.0/ . Moody GB, Mark RG. The impact of the MIT-BIH arrhythmia database. IEEE Eng Med Biol Mag 2001;20(3):45-50. Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation 2000;101(23):e215-e220.
2. **St. Petersburg INCART 12-lead Arrhythmia Database** (v1.0.0), https://physionet.org/content/incartdb/1.0.0/ , doi:10.13026/C2V88N. Goldberger AL et al., Circulation 2000 (as above).

Licence: both databases are distributed by PhysioNet under the Open Data Commons Attribution License v1.0 (ODC-By 1.0) according to their PhysioNet dataset pages (read 2026-11-19); attribution is required. Derived arrays in this archive (per-beat logits, labels, patient/record identifiers, beat indices) contain no signal samples.

Processed beat arrays (`mitbih_beats_r2.npz`, `incart_beats_r2.npz`) are not redistributed; regenerate them with `preprocessing/04_extract_beats.py` (hashes in `manifests/processed_data_hashes.json`). INCART patient grouping uses `files-patients-diagnoses.txt` from the PhysioNet download.
