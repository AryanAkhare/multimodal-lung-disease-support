import os
import io
import json
import tempfile
from pathlib import Path
from typing import Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import librosa
import joblib
from PIL import Image

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
    primary_finding: str
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
        """Load the global model and preprocessors"""
        model_dir = Path(__file__).parent.parent / "models" / "keras"
        
        # Try loading keras format first, then h5 format
        keras_model = model_dir / "best_global_model.keras"
        h5_model = model_dir / "best_global_model.h5"
        
        if keras_model.exists():
            print(f"Loading model from {keras_model}")
            self.model = keras.models.load_model(str(keras_model))
        elif h5_model.exists():
            print(f"Loading model from {h5_model}")
            self.model = keras.models.load_model(str(h5_model))
        else:
            raise FileNotFoundError(
                f"Global model not found. Checked:\n- {keras_model}\n- {h5_model}"
            )
        
        # Load preprocessors
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
                print(f"Loaded {key} preprocessor")
            else:
                print(f"Warning: {filename} not found")


# ==================== PREPROCESSING FUNCTIONS ====================

def preprocess_image(image_data: bytes, target_size: tuple = (224, 224)) -> np.ndarray:
    """Preprocess image for model input"""
    image = Image.open(io.BytesIO(image_data))
    image = image.convert('RGB')
    image = np.array(image.resize(target_size))
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=0)


def preprocess_audio(audio_data: bytes, sr: int = 22050, duration: float = 5.0) -> np.ndarray:
    """Preprocess audio for model input"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name
    
    try:
        # Load audio
        y, sr_loaded = librosa.load(tmp_path, sr=sr, duration=duration)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # Ensure consistent shape
        target_time_steps = int(sr * duration // 512)  # Approximate
        if mfcc.shape[1] < target_time_steps:
            mfcc = np.pad(mfcc, ((0, 0), (0, target_time_steps - mfcc.shape[1])))
        else:
            mfcc = mfcc[:, :target_time_steps]
        
        return np.expand_dims(mfcc, axis=0)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def preprocess_tabular(inputs: TabularInputs, loader: ModelLoader) -> np.ndarray:
    """Preprocess tabular data for model input"""
    # Create feature array
    categorical_features = {
        'gender': inputs.gender,
        'smoking_status': inputs.smoking_status,
        'wheeze': inputs.wheeze,
        'chest_tightness': inputs.chest_tightness,
        'crackles': inputs.crackles,
        'fever': inputs.fever,
    }
    
    numerical_features = {
        'age': inputs.age,
        'fev1_percent': inputs.fev1_percent,
        'spo2': inputs.spo2,
        'respiratory_rate': inputs.respiratory_rate,
        'cough_severity': inputs.cough_severity,
        'bmi': inputs.bmi,
        'copd_exacerbations': inputs.copd_exacerbations,
    }
    
    # Prepare data for preprocessing
    data = np.array([list(categorical_features.values()) + list(numerical_features.values())])
    
    # Apply preprocessors if available
    if 'cat_imputer' in loader.preprocessors:
        cat_data = np.array([list(categorical_features.values())])
        cat_imputer = loader.preprocessors['cat_imputer']
        cat_data = cat_imputer.transform(cat_data)
    else:
        cat_data = np.array([list(categorical_features.values())])
    
    if 'num_imputer' in loader.preprocessors:
        num_data = np.array([list(numerical_features.values())])
        num_imputer = loader.preprocessors['num_imputer']
        num_data = num_imputer.transform(num_data)
    else:
        num_data = np.array([list(numerical_features.values())])
    
    if 'scaler' in loader.preprocessors:
        scaler = loader.preprocessors['scaler']
        num_data = scaler.transform(num_data)
    
    # Combine categorical and numerical
    processed_data = np.hstack([cat_data, num_data])
    return processed_data.astype('float32')


# ==================== PREDICTION FUNCTIONS ====================

def get_disease_label_and_message(prediction: float) -> tuple:
    """Convert prediction confidence to label and message"""
    threshold = 0.5
    if prediction >= threshold:
        label = "Disease Detected"
        message = f"High disease probability ({prediction*100:.1f}%)"
    else:
        label = "Healthy/Low Risk"
        message = f"Low disease probability ({prediction*100:.1f}%)"
    return label, message


def generate_final_report(
    image_pred: float,
    audio_pred: float,
    tabular_pred: float,
    provided_modalities: list
) -> FinalReport:
    """Generate comprehensive final report based on all predictions"""
    
    # Calculate overall confidence
    predictions = []
    if 'image' in provided_modalities:
        predictions.append(image_pred)
    if 'audio' in provided_modalities:
        predictions.append(audio_pred)
    if 'tabular' in provided_modalities:
        predictions.append(tabular_pred)
    
    overall_confidence = np.mean(predictions) if predictions else 0.5
    
    # Determine primary finding
    if overall_confidence >= 0.7:
        primary_finding = "High likelihood of lung disease"
        severity = "High"
    elif overall_confidence >= 0.4:
        primary_finding = "Moderate disease indicators present"
        severity = "Moderate"
    else:
        primary_finding = "Low disease indicators - likely healthy"
        severity = "Low"
    
    # Generate recommendations
    recommendations = []
    if severity == "High":
        recommendations = [
            "Immediate consultation with a pulmonologist recommended",
            "Consider advanced imaging tests (CT scan)",
            "Monitor respiratory symptoms closely",
            "Follow up within 48 hours"
        ]
    elif severity == "Moderate":
        recommendations = [
            "Schedule consultation with a healthcare provider",
            "Monitor symptoms for next 1-2 weeks",
            "Consider additional tests if symptoms worsen",
            "Follow up within 1 week"
        ]
    else:
        recommendations = [
            "Continue regular health monitoring",
            "Maintain healthy lifestyle habits",
            "Annual check-ups recommended",
            "Report any new respiratory symptoms"
        ]
    
    note = f"This is an AI-assisted analysis and should not replace professional medical diagnosis. All findings are based on multimodal data analysis."
    
    return FinalReport(
        primary_finding=primary_finding,
        overall_confidence=round(overall_confidence, 4),
        recommendation=recommendations,
        note=note
    )


# ==================== FASTAPI APP ====================

# Global model loader
model_loader = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle"""
    global model_loader
    model_loader = ModelLoader()
    print("✅ FastAPI Server Started - Model Loaded")
    yield
    print("🛑 FastAPI Server Shutdown")


app = FastAPI(
    title="Multimodal Lung Disease Detection API",
    description="API for lung disease detection using image, audio, and tabular data",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loader is not None and model_loader.model is not None
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    age: float = Form(...),
    gender: str = Form(...),
    smoking_status: str = Form(...),
    fev1_percent: float = Form(...),
    spo2: float = Form(...),
    respiratory_rate: float = Form(...),
    cough_severity: float = Form(...),
    wheeze: str = Form(...),
    chest_tightness: str = Form(...),
    crackles: str = Form(...),
    fever: str = Form(...),
    bmi: float = Form(...),
    copd_exacerbations: float = Form(...),
):
    """
    Make predictions on multimodal patient data
    """
    if not image and not audio:
        raise HTTPException(
            status_code=400,
            detail="At least image or audio must be provided"
        )
    
    try:
        # Parse tabular inputs
        tabular_inputs = TabularInputs(
            age=age,
            gender=gender,
            smoking_status=smoking_status,
            fev1_percent=fev1_percent,
            spo2=spo2,
            respiratory_rate=respiratory_rate,
            cough_severity=cough_severity,
            wheeze=wheeze,
            chest_tightness=chest_tightness,
            crackles=crackles,
            fever=fever,
            bmi=bmi,
            copd_exacerbations=copd_exacerbations,
        )
        
        # Track provided modalities
        provided_modalities = []
        
        # Process image
        image_pred = 0.1
        if image:
            provided_modalities.append('image')
            image_data = await image.read()
            image_input = preprocess_image(image_data)
            image_output = model_loader.model.predict(image_input, verbose=0)
            image_pred = float(image_output[0][0]) if isinstance(image_output, np.ndarray) else 0.1
        
        # Process audio
        audio_pred = 0.1
        if audio:
            provided_modalities.append('audio')
            audio_data = await audio.read()
            audio_input = preprocess_audio(audio_data)
            audio_output = model_loader.model.predict(audio_input, verbose=0)
            audio_pred = float(audio_output[0][0]) if isinstance(audio_output, np.ndarray) else 0.1
        
        # Process tabular
        provided_modalities.append('tabular')
        tabular_input = preprocess_tabular(tabular_inputs, model_loader)
        tabular_output = model_loader.model.predict(tabular_input, verbose=0)
        tabular_pred = float(tabular_output[0][0]) if isinstance(tabular_output, np.ndarray) else 0.1
        
        # Get predictions within [0, 1] range
        image_pred = np.clip(image_pred, 0, 1)
        audio_pred = np.clip(audio_pred, 0, 1)
        tabular_pred = np.clip(tabular_pred, 0, 1)
        
        # Create branch scores
        image_label, image_msg = get_disease_label_and_message(image_pred)
        audio_label, audio_msg = get_disease_label_and_message(audio_pred)
        tabular_label, tabular_msg = get_disease_label_and_message(tabular_pred)
        
        image_branch = BranchScore(
            label=image_label,
            confidence=round(image_pred, 4),
            message=image_msg
        )
        
        audio_branch = BranchScore(
            label=audio_label,
            confidence=round(audio_pred, 4),
            message=audio_msg
        )
        
        tabular_branch = BranchScore(
            label=tabular_label,
            confidence=round(tabular_pred, 4),
            message=tabular_msg
        )
        
        # Generate final report
        final_report = generate_final_report(
            image_pred, audio_pred, tabular_pred, provided_modalities
        )
        
        return PredictionResponse(
            image_branch=image_branch,
            audio_branch=audio_branch,
            tabular_branch=tabular_branch,
            final_report=final_report
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Multimodal Lung Disease Detection API",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
