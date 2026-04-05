from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .config import Settings
from .schemas import PredictionResponse
from .model_stub import evaluate_prediction

settings = Settings()
app = FastAPI(title='Multimodal Lung Disease Support API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:4173'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/api/predict', response_model=PredictionResponse)
async def predict(
    image: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
    age: int = Form(...),
    gender: str = Form(...),
    smoking_status: str = Form(...),
    fev1_percent: float = Form(...),
    spo2: float = Form(...),
    respiratory_rate: float = Form(...),
    cough_severity: int = Form(...),
    wheeze: str = Form(...),
    chest_tightness: str = Form(...),
    crackles: str = Form(...),
    fever: str = Form(...),
    bmi: float = Form(...),
    copd_exacerbations: int = Form(...),
):
    """Main prediction endpoint: processes multimodal inputs and returns structured report."""
    if image is None and audio is None:
        raise HTTPException(status_code=422, detail='At least one modality file is required.')

    inputs = {
        'image': image,
        'audio': audio,
        'tabular': {
            'age': age,
            'gender': gender,
            'smoking_status': smoking_status,
            'fev1_percent': fev1_percent,
            'spo2': spo2,
            'respiratory_rate': respiratory_rate,
            'cough_severity': cough_severity,
            'wheeze': wheeze,
            'chest_tightness': chest_tightness,
            'crackles': crackles,
            'fever': fever,
            'bmi': bmi,
            'copd_exacerbations': copd_exacerbations,
        }
    }

    try:
        result = await evaluate_prediction(inputs, settings)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Prediction failed: {str(e)}')
