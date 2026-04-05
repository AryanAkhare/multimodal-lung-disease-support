# Documentation

## Project summary

This project started as an attempt to build one fully fused multimodal classifier from image, audio, and tabular public datasets. After running the notebook and inspecting the real results, the final project design was refined into a more defensible 3-model decision-support system.

The notebook used is [Multimodal_Colab_Ready.ipynb](/c:/Users/aryan/Desktop/Subjects/Coding/COPD/Multimodal_Colab_Ready.ipynb).

## Final model design

### 1. Image binary model

- file: `image_binary_model.keras`
- role: primary branch
- task: `healthy` vs `disease`

Observed result:

- test accuracy about `0.9603`
- test F1 about `0.9593`

This is the strongest branch and should be presented as the main disease-detection signal.

### 2. Audio binary model

- file: `audio_binary_model.keras`
- role: auxiliary branch
- task: `healthy` vs `disease`

Observed result:

- unstable
- weak generalization
- final performance too poor to trust as a headline signal

Reason:

- the healthy class in the respiratory-sound dataset is extremely small compared with disease classes
- the branch overfits or collapses easily

Conclusion:

- keep this branch in the project as an experimental auxiliary modality
- do not use it as a primary decision driver

### 3. Tabular profile model

- file: `tabular_profile_model.keras`
- role: supportive branch
- task: multiclass disease-profile suggestion

Observed result:

- weak exact-class prediction performance
- still useful as profile resemblance / disease-pattern context

Conclusion:

- do not present it as a standalone diagnosis model
- use it to suggest likely disease-pattern matches such as COPD, bronchitis, or pneumonia

## Final report design

The final doctor-style report should contain:

- `headline`
- `binary_signals`
- `tabular_profile`
- `clinical_note`

Recommended weighting:

- image branch: highest trust
- audio branch: lowest trust
- tabular branch: supportive context

Recommended wording:

- image is `primary`
- audio is `auxiliary`
- tabular is `supportive`

## Example interpretation of final report

When the report shows:

- very low image disease probability
- weak / unstable audio probability
- tabular profile leaning toward `copd`, `bronchitis`, or `pneumonia`

the correct interpretation is:

- the main imaging evidence does not strongly support disease in that sample
- audio is not stable enough to override imaging
- tabular suggests what patterns are most similar if disease-context features are present

This is exactly why the report should be framed as support for review, not diagnosis.

## Why the project is still valid academically

Even though not every branch is equally strong, the project is still valid and useful academically because:

- it is genuinely multimodal
- it uses the provided datasets honestly
- it reflects real-world engineering tradeoffs
- it produces a coherent doctor-facing output
- it avoids overclaiming clinical validity

## How many models are created

The final system creates **3 trained models**:

1. `image_binary_model.keras`
2. `audio_binary_model.keras`
3. `tabular_profile_model.keras`

## Final presentation guidance

Describe the system as:

- a multimodal decision-support prototype
- a doctor-assist reporting pipeline
- an academic screening and profile-suggestion system

Do not describe it as:

- an automatic diagnostic system
- a clinically validated tool
- a final medical decision maker

## Artifact locations

Generated in Colab under:

- `/content/lung_disease_multimodal_project/artifacts`
- `/content/lung_disease_multimodal_project/reports`

Important outputs:

- `image_binary_model.keras`
- `audio_binary_model.keras`
- `tabular_profile_model.keras`
- `metrics_summary.json`
- `sample_doctor_report.json`
