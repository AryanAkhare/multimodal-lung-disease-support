UI
Your UI should feel like a doctor-assist form, not a consumer app.

Inputs to upload:

chest image
respiratory audio file
tabular/clinical form fields
Tabular form can include:

age
gender
smoking status
lung capacity
treatment type
hospital visits
any other fields used by the saved tabular schema
Recommended layout:

left side: input panel
right side: report panel
Input panel:

image uploader
audio uploader
tabular form
Generate Report button
Output panel:

headline: Low disease support or Disease support detected
primary signal: image disease probability
auxiliary signal: audio disease probability
supportive profile: top 3 tabular pattern suggestions
clinical note: “support tool, not diagnosis”
Expected output shape:

image disease probability
audio disease probability
combined disease support
top tabular profile suggestions like COPD, Bronchitis, Pneumonia
roles:
primary
auxiliary
supportive
So the project expects:

3 inputs: image + audio + tabular
1 combined doctor-style report as output