# Redaction note

Two manifests recorded a user-profile directory (`C:\Users\<user>\AppData\Roaming\Python\Python313\site-packages`) as the location of the numpy package. To avoid publishing a local user path, that single string was replaced by `<user-site-packages>` in the copies in this archive. Nothing else was changed. The originals are retained unmodified in the project's frozen result tree.

| file | SHA-256 of the original (project tree) | SHA-256 of the redacted copy (this archive) |
|---|---|---|
| manifests/manifest_pre_resume.json | 6d564a8cb46fb89c99ad76e2781acd457bb7f63353346ab75311a62d0ca3f563 | 8ffe0bc35dc95e56c72b456c71464d5471d9050b5d0f67e8094bae19df573e9b |
| manifests/manifest_post_training.json | ea03044b4c228f628d5a29b88531159dff33c1b825257ba71faccad5180939bb | c55fbdad2a96c29a3f2adc42e49126d678c3d4231d79caf701f3e8c3134e712f |

`final_results_manifest.json` lists the hash of the ORIGINAL `manifest_post_training.json` and therefore differs from the redacted copy by design. All other manifest entries (scripts, scores, evaluation outputs, checkpoints) refer to files copied byte-identically.

Other local-path constants remain in the frozen scripts and manifests (`D:\explainable_ai\...`, the MIT-BIH and ECGTransForm folders); they were left unmodified so that the recorded file hashes verify. See README.md ("Running the code") for the constants that must be edited.
