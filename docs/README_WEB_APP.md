# README

This project is a multimodal academic decision-support prototype built from the Colab notebook [Multimodal_Colab_Ready.ipynb](/c:/Users/aryan/Desktop/Subjects/Coding/COPD/Multimodal_Colab_Ready.ipynb).

## Final architecture

The final system uses 3 trained models:

1. `image_binary_model.keras`
   - main disease detector
   - task: `healthy` vs `disease`

2. `audio_binary_model.keras`
   - auxiliary experimental detector
   - task: `healthy` vs `disease`

3. `tabular_profile_model.keras`
   - supportive disease-pattern model
   - task: multiclass disease-profile suggestion

The final report combines them into a doctor-style summary.

## Actual observed branch quality

### Image branch

- strongest branch
- test accuracy: about `0.9603`
- test F1: about `0.9593`
- use this as the primary decision signal

### Audio branch

- weakest branch
- highly unstable due to very small healthy-audio class
- should be treated as auxiliary only
- in the final report it can be suppressed from the headline when unstable

### Tabular branch

- not reliable as a standalone disease classifier
- useful as supportive disease-pattern context
- should be shown as profile resemblance, not diagnosis

## Final report behavior

The report should emphasize:

- image as `primary`
- audio as `auxiliary`
- tabular as `supportive`

Recommended user-facing interpretation:

- image: main disease-support signal
- audio: secondary / low-confidence cue
- tabular: likely profile patterns to review

## Colab output files

The notebook writes outputs under:

- `/content/lung_disease_multimodal_project/artifacts`
- `/content/lung_disease_multimodal_project/reports`

Important files:

- `image_binary_model.keras`
- `audio_binary_model.keras`
- `tabular_profile_model.keras`
- `tabular_num_imputer.joblib`
- `tabular_cat_imputer.joblib`
- `tabular_scaler.joblib`
- `tabular_schema.joblib`
- `tabular_label_encoder.joblib`
- `metrics_summary.json`
- `sample_doctor_report.json`

## Web-app recommendation

For the project app:

- show image disease probability prominently
- show audio as experimental auxiliary evidence
- show top tabular profile suggestions below that
- include a clear note that this supports review rather than replacing diagnosis
