# Legacy research demonstration (not included in this release)

The project folder `clinical_ai/` contains a legacy, **unvalidated** software demonstration of a report-generating interface. It is **not connected to the evaluated PyTorch models** of this study: it uses a separate TensorFlow/Keras model that takes 300-sample windows and different preprocessing (no R-peak alignment, no db5 denoising, no resampling to 186 samples, no sampling-rate check).
Its attribution routines, score aggregation and flagging logic have not been validated. It does **not** establish clinical utility, clinical decision support, triage capability or validated deployment, and it must not be used for any clinical purpose. No result of this demonstration is reported in the associated article.
This notice replaces the earlier wording that described the demonstration as a "clinical decision support prototype". The demonstration code itself is excluded from this release.
