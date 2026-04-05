# Backend - Multimodal Lung Disease Detection API

A FastAPI-based inference server for multimodal lung disease detection. Loads the pre-trained global model and handles predictions on image, audio, and tabular patient data.

## Features

- ✅ Loads `best_global_model.keras` or `best_global_model.h5`
- ✅ Processes X-ray images, respiratory audio, and clinical tabular features
- ✅ Returns structured multimodal predictions with confidence scores
- ✅ CORS-enabled for frontend integration
- ✅ Comprehensive error handling and validation

## Prerequisites

- Python 3.8+
- TensorFlow 2.21+
- All dependencies in `requirements.txt`

## Setup

### 1. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Backend

From the project root directory:

```bash
# Windows
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# macOS/Linux
python3 -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Or directly run:
```bash
python backend/main.py
```

### Expected Output
```
INFO:     Uvicorn running on http://0.0.0.0:8000
✅ FastAPI Server Started - Model Loaded
```

## API Endpoints

### 1. Health Check
```bash
GET http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Predict (Multimodal Inference)
```bash
POST http://localhost:8000/api/predict
```

**Form Data:**
- `image` (optional): Chest X-ray image file (PNG/JPG)
- `audio` (optional): Respiratory audio recording (WAV/MP3)
- `age`: Patient age (number)
- `gender`: Patient gender (string: "Male"/"Female"/"Other")
- `smoking_status`: Smoking status (string: "Never"/"Former"/"Current")
- `fev1_percent`: FEV1 percentage (number)
- `spo2`: Blood oxygen saturation (number: 0-100)
- `respiratory_rate`: Respiratory rate in breaths per minute (number)
- `cough_severity`: Cough severity (number: 0-5)
- `wheeze`: Presence of wheezing (string: "Yes"/"No")
- `chest_tightness`: Chest tightness (string: "Yes"/"No")
- `crackles`: Presence of crackles (string: "Yes"/"No")
- `fever`: Presence of fever (string: "Yes"/"No")
- `bmi`: Body Mass Index (number)
- `copd_exacerbations`: Number of COPD exacerbations (number)

**Response:**
```json
{
  "image_branch": {
    "label": "Disease Detected",
    "confidence": 0.7543,
    "message": "High disease probability (75.43%)"
  },
  "audio_branch": {
    "label": "Healthy/Low Risk",
    "confidence": 0.2341,
    "message": "Low disease probability (23.41%)"
  },
  "tabular_branch": {
    "label": "Disease Detected",
    "confidence": 0.6234,
    "message": "High disease probability (62.34%)"
  },
  "final_report": {
    "primary_finding": "Moderate disease indicators present",
    "overall_confidence": 0.5373,
    "recommendation": [
      "Schedule consultation with a healthcare provider",
      "Monitor symptoms for next 1-2 weeks",
      "Consider additional tests if symptoms worsen",
      "Follow up within 1 week"
    ],
    "note": "This is an AI-assisted analysis and should not replace professional medical diagnosis. All findings are based on multimodal data analysis."
  }
}
```

## Model Requirements

The backend expects the following model files in the `models/` directory:

**Primary Model (Required):**
- `models/keras/best_global_model.keras` (preferred)
- OR `models/keras/best_global_model.h5` (fallback)

**Preprocessors (Optional but Recommended):**
- `artifacts/tabular_scaler.joblib`
- `artifacts/tabular_num_imputer.joblib`
- `artifacts/tabular_cat_imputer.joblib`
- `artifacts/tabular_label_encoder.joblib`

## Testing

### Using cURL
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -F "age=38" \
  -F "gender=Male" \
  -F "smoking_status=Never" \
  -F "fev1_percent=72" \
  -F "spo2=96" \
  -F "respiratory_rate=18" \
  -F "cough_severity=2" \
  -F "wheeze=No" \
  -F "chest_tightness=No" \
  -F "crackles=No" \
  -F "fever=No" \
  -F "bmi=24.5" \
  -F "copd_exacerbations=0" \
  -F "image=@path/to/xray.jpg" \
  -F "audio=@path/to/audio.wav"
```

### Using Python Requests
```python
import requests

url = "http://localhost:8000/api/predict"
files = {
    'image': open('xray.jpg', 'rb'),
    'audio': open('audio.wav', 'rb'),
}
data = {
    'age': 38,
    'gender': 'Male',
    'smoking_status': 'Never',
    'fev1_percent': 72,
    'spo2': 96,
    'respiratory_rate': 18,
    'cough_severity': 2,
    'wheeze': 'No',
    'chest_tightness': 'No',
    'crackles': 'No',
    'fever': 'No',
    'bmi': 24.5,
    'copd_exacerbations': 0,
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

### Using Postman
1. Create POST request to `http://localhost:8000/api/predict`
2. Under "Body" tab, select "form-data"
3. Add all fields (image, audio, and tabular data)
4. Send the request

## Troubleshooting

### Model Not Found
```
FileNotFoundError: Global model not found. Checked:
```
**Solution:** Ensure `best_global_model.keras` or `best_global_model.h5` exists in the `models/keras/` directory.

### CORS Errors
The API is configured with CORS to accept requests from:
- `http://localhost:5173` (Vite frontend)
- `http://localhost:3000` (Default Node/React port)
- `*` (All origins - for development)

For production, update the CORS origins in `main.py`:
```python
allow_origins=["https://yourdomain.com"],
```

### Out of Memory
Reduce model batch size or increase system RAM. TensorFlow models typically require 2-4GB VRAM.

### Port Already in Use
```bash
# Change port to 8001
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001
```

## Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Deployment

For production deployment:
1. Set `--reload=False`
2. Use a production ASGI server like Gunicorn:
```bash
pip install gunicorn
gunicorn backend.main:app -w 4 -b 0.0.0.0:8000
```

## Notes

- At least one input (image or audio) must be provided for prediction
- Tabular data is optional but enhances prediction accuracy
- Predictions return confidence scores between 0 and 1
- All predictions should be reviewed by medical professionals
