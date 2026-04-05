import os
import io
import tempfile
from pathlib import Path
from typing import Optional
import numpy as np
import tensorflow as tf
import keras
import cv2
import librosa
import joblib
from PIL import Image
import uvicorn

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# ==================== DATA MODELS ====================

class TabularInputs(BaseModel):
    age: float
    gender: str
    smoking_status: str
    fev1_percent: float
    spo2: float
    respiratory_rate: float
    cough_severity: float
    wheeze: str
    chest_tightness: str
    crackles: str
    fever: str
    bmi: float
    copd_exacerbations: float

class BranchScore(BaseModel):
    label: str
    confidence: float
    message: str

class FinalReport(BaseModel):
    predicted_class: str
    overall_confidence: float
    recommendation: list
    note: str

class PredictionResponse(BaseModel):
    image_branch: BranchScore
    audio_branch: BranchScore
    tabular_branch: BranchScore
    final_report: FinalReport

# ==================== MODEL LOADER ====================

class ModelLoader:
    def __init__(self):
        self.model = None
        self.preprocessors = {}
        self.load_models()

    def load_models(self):
        model_dir = Path(__file__).parent.parent / "models" / "keras"
        keras_model = model_dir / "best_global_model.keras"
        h5_model = model_dir / "best_global_model.h5"
        
        if keras_model.exists():
            self.model = keras.models.load_model(str(keras_model))
        elif h5_model.exists():
            self.model = keras.models.load_model(str(h5_model))
        else:
            raise FileNotFoundError("Global model not found.")
        
        artifacts_dir = Path(__file__).parent.parent / "artifacts"
        preprocessor_files = {
            'scaler': 'tabular_scaler.joblib',
            'num_imputer': 'tabular_num_imputer.joblib',
            'cat_imputer': 'tabular_cat_imputer.joblib',
            'label_encoder': 'tabular_label_encoder.joblib',
        }
        
        for key, filename in preprocessor_files.items():
            filepath = artifacts_dir / filename
            if filepath.exists():
                self.preprocessors[key] = joblib.load(str(filepath))

# ==================== PREPROCESSING FUNCTIONS ====================

def preprocess_image(image_data: bytes, target_size: tuple = (224, 224)) -> np.ndarray:
    image = Image.open(io.BytesIO(image_data))
    image = image.convert('RGB')
    image = np.array(image.resize(target_size))
    image = keras.applications.densenet.preprocess_input(image.astype(np.float32))
    return np.expand_dims(image, axis=0)

def preprocess_audio(audio_data: bytes, sr: int = 22050, duration: float = 5.0, target_size: tuple = (224, 224)) -> np.ndarray:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name
    try:
        y, sr_loaded = librosa.load(tmp_path, sr=sr, duration=duration, mono=True)
        target_len = int(sr * duration)
        y = np.pad(y, (0, max(0, target_len - len(y))), mode="constant")[:target_len]

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=2000)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        mel_n = ((mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8) * 255).astype(np.uint8)
        mel_r = cv2.resize(mel_n, target_size, interpolation=cv2.INTER_LINEAR)

        img = np.stack([mel_r] * 3, axis=-1).astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        print(f"Audio processing failed: {e}")
        return np.zeros((1, *target_size, 3), dtype=np.float32)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def preprocess_tabular(inputs: TabularInputs, loader: ModelLoader) -> np.ndarray:
    gender_enc = 1.0 if inputs.gender.upper() == 'MALE' else 0.0
    smoking_map = {'Never': 0.0, 'Former': 1.0, 'Current': 2.0}
    smoking_status = smoking_map.get(inputs.smoking_status, 0.0)
    
    yes_no_map = {'Yes': 1.0, 'No': 0.0}
    wheeze = yes_no_map.get(inputs.wheeze, 0.0)
    chest_tightness = yes_no_map.get(inputs.chest_tightness, 0.0)
    crackles = yes_no_map.get(inputs.crackles, 0.0)
    fever = yes_no_map.get(inputs.fever, 0.0)
    
    feature_vector = [
        float(inputs.age), float(gender_enc), float(smoking_status),
        float(inputs.fev1_percent), float(inputs.spo2), float(inputs.respiratory_rate),
        float(inputs.cough_severity), float(wheeze), float(chest_tightness),
        float(crackles), float(fever), float(inputs.bmi), float(inputs.copd_exacerbations)
    ]
    
    data_array = np.array([feature_vector], dtype=np.float32)
    if 'scaler' in loader.preprocessors:
        try:
            data_array = loader.preprocessors['scaler'].transform(data_array)
        except Exception as e:
            print(f"Warning: Scaler transform failed. Error: {e}")
            
    return data_array.astype('float32')

# ==================== PREDICTION FUNCTIONS ====================

def get_disease_label_and_message(prediction: float) -> tuple:
    if prediction >= 0.5:
        return "Disease Detected", f"High disease probability ({prediction*100:.1f}%)"
    return "Healthy/Low Risk", f"Low disease probability ({prediction*100:.1f}%)"

def generate_final_report(probs: np.ndarray) -> FinalReport:
    MASTER_CLASSES = ["asthma", "copd", "healthy", "pneumonia"]
    
    idx = int(np.argmax(probs))
    predicted_class = MASTER_CLASSES[idx]
    confidence = float(probs[idx])
    
    recommendations = {
        "copd": ["Spirometry (FEV1/FVC < 0.70)", "CT chest for emphysema", "Initiate LABA/LAMA"],
        "pneumonia": ["Chest X-ray for consolidation", "CBC, CRP monitoring", "Empiric antibiotics"],
        "asthma": ["Peak flow monitoring", "IgE testing", "Initiate ICS step therapy"],
        "healthy": ["No acute respiratory pathology detected", "Annual routine check-up", "Maintain healthy lifestyle"]
    }
    
    return FinalReport(
        predicted_class=predicted_class.upper(),
        overall_confidence=confidence,
        recommendation=recommendations.get(predicted_class, []),
        note="AI-synthesized multimodal evidence."
    )

# ==================== FASTAPI APP ====================

model_loader = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_loader
    model_loader = ModelLoader()
    print("✅ FastAPI Server Started")
    yield

app = FastAPI(title="Multimodal Lung Disease Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(
    image: Optional[UploadFile] = File(None), audio: Optional[UploadFile] = File(None),
    age: float = Form(...), gender: str = Form(...), smoking_status: str = Form(...),
    fev1_percent: float = Form(...), spo2: float = Form(...), respiratory_rate: float = Form(...),
    cough_severity: float = Form(...), wheeze: str = Form(...), chest_tightness: str = Form(...),
    crackles: str = Form(...), fever: str = Form(...), bmi: float = Form(...),
    copd_exacerbations: float = Form(...),
):
    if not image and not audio:
        raise HTTPException(status_code=400, detail="At least image or audio must be provided")
    
    try:
        tabular_inputs = TabularInputs(
            age=age, gender=gender, smoking_status=smoking_status, fev1_percent=fev1_percent,
            spo2=spo2, respiratory_rate=respiratory_rate, cough_severity=cough_severity,
            wheeze=wheeze, chest_tightness=chest_tightness, crackles=crackles, fever=fever,
            bmi=bmi, copd_exacerbations=copd_exacerbations
        )
        
        image_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        audio_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        
        if image:
            image_input = preprocess_image(await image.read())
        if audio:
            audio_input = preprocess_audio(await audio.read())
            
        tabular_input = preprocess_tabular(tabular_inputs, model_loader)
        
        predictions = model_loader.model.predict([image_input, audio_input, tabular_input], verbose=0)[0]
        
        disease_prob = 1.0 - float(predictions[2])
        disease_prob = np.clip(disease_prob, 0, 1)

        img_score = disease_prob if image else 0.0
        aud_score = disease_prob if audio else 0.0
        
        image_label, image_msg = get_disease_label_and_message(img_score)
        audio_label, audio_msg = get_disease_label_and_message(aud_score)
        tabular_label, tabular_msg = get_disease_label_and_message(disease_prob)

        return PredictionResponse(
            image_branch=BranchScore(label=image_label if image else "Not Provided", confidence=img_score, message=image_msg if image else "-"),
            audio_branch=BranchScore(label=audio_label if audio else "Not Provided", confidence=aud_score, message=audio_msg if audio else "-"),
            tabular_branch=BranchScore(label=tabular_label, confidence=disease_prob, message=tabular_msg),
            final_report=generate_final_report(predictions)
        )
        
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")