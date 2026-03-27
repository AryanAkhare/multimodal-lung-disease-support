# Prompt

Use this file to restore the project context in future sessions.

## Project identity

This is an academic multimodal lung-disease decision-support prototype.

Main notebook:

- [Multimodal_Colab_Ready.ipynb](/c:/Users/aryan/Desktop/Subjects/Coding/COPD/Multimodal_Colab_Ready.ipynb)

Markdown docs:

- [README_WEB_APP.md](/c:/Users/aryan/Desktop/Subjects/Coding/COPD/markdown/README_WEB_APP.md)
- [Documentation.md](/c:/Users/aryan/Desktop/Subjects/Coding/COPD/markdown/Documentation.md)
- [Prompt.md](/c:/Users/aryan/Desktop/Subjects/Coding/COPD/markdown/Prompt.md)

## Datasets used

Kaggle datasets used in Colab:

1. `samikshadalvi/lungs-diseases-dataset`
2. `vbookshelf/respiratory-sound-database`
3. `omkarmanohardalvi/lungs-disease-dataset-4-types`

## Final architecture

There are **3 trained models**:

1. Image binary model
   - `image_binary_model.keras`
   - `healthy` vs `disease`
   - primary branch

2. Audio binary model
   - `audio_binary_model.keras`
   - `healthy` vs `disease`
   - auxiliary / experimental branch

3. Tabular profile model
   - `tabular_profile_model.keras`
   - multiclass disease-pattern suggestion
   - supportive branch

## Final interpretation rules

- image branch is the most trustworthy signal
- audio branch is weak and should not dominate conclusions
- tabular branch is supportive context, not final diagnosis
- final report must be framed as decision support, not diagnosis

## Important measured results

Image branch:

- accuracy about `0.9603`
- F1 about `0.9593`
- strongest branch

Audio branch:

- unstable due to extreme class imbalance
- treat as auxiliary

Tabular branch:

- weak as exact disease classifier
- useful as disease-profile resemblance only

## Final report structure

Expected report sections:

- `headline`
- `binary_signals`
- `tabular_profile`
- `clinical_note`

Roles:

- image = `primary`
- audio = `auxiliary`
- tabular = `supportive`

## What not to change casually

- do not reframe the system as clinically validated
- do not promote audio to the main branch without better data
- do not use tabular as the main diagnosis engine

## Best future direction

If working on this again, prioritize:

1. building the web app
2. polishing report UX and wording
3. optionally simplifying or demoting the audio branch further
