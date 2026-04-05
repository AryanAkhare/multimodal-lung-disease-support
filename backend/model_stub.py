from typing import Any
from pathlib import Path
import numpy as np
import joblib
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    tf = None
    keras = None

from .config import Settings
from .schemas import PredictionResponse, BranchScore, FinalReport

# Global model cache
_model_cache = {}
_tabular_scaler = None

def get_tabular_scaler():
    """Load tabular StandardScaler once at startup."""
    global _tabular_scaler
    if _tabular_scaler is None:
        try:
            _tabular_scaler = joblib.load('artifacts/tabular_scaler.joblib')
        except Exception as e:
            print(f"Warning: Failed to load tabular scaler: {e}")
    return _tabular_scaler

def load_model(model_path: str):
    """Load a model from disk with caching."""
    if model_path in _model_cache:
        return _model_cache[model_path]
    
    if not Path(model_path).exists():
        return None
    
    try:
        if model_path.endswith('.keras') or model_path.endswith('.h5'):
            model = keras.models.load_model(model_path)
        else:
            return None
        _model_cache[model_path] = model
        return model
    except Exception as e:
        print(f"Failed to load model {model_path}: {e}")
        return None

async def evaluate_prediction(inputs: dict[str, Any], settings: Settings) -> PredictionResponse:
    # Load models and evaluate branches
    image_score = await evaluate_image_branch(inputs.get('image'), settings)
    audio_score = await evaluate_audio_branch(inputs.get('audio'), settings)
    tabular_score = await evaluate_tabular_branch(inputs.get('tabular'), settings)

    final_report = build_support_report(image_score, audio_score, tabular_score)

    return PredictionResponse(
        image_branch=image_score,
        audio_branch=audio_score,
        tabular_branch=tabular_score,
        final_report=final_report,
    )

async def evaluate_image_branch(image_file: Any, settings: Settings) -> BranchScore:
    if image_file is None:
        return BranchScore(
            label='Unavailable',
            confidence=0.0,
            message='No imaging data received. Primary modality unavailable.'
        )

    try:
        import io
        from PIL import Image
        
        content = await image_file.read()
        image = Image.open(io.BytesIO(content)).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        model = load_model(settings.image_model_path)
        if model is None:
            return BranchScore(label='Error', confidence=0.0, message='Image model unavailable.')
        
        prediction = model.predict(image_array, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction) * 100)
        
        classes = ["asthma", "copd", "healthy", "pneumonia"]
        label = classes[class_idx].capitalize()
        
        return BranchScore(label=label, confidence=confidence, message=f'Chest imaging: {label} ({confidence:.1f}%)')
    except Exception as e:
        return BranchScore(label='Error', confidence=0.0, message=f'Image processing failed: {str(e)}')

async def evaluate_audio_branch(audio_file: Any, settings: Settings) -> BranchScore:
    if audio_file is None:
        return BranchScore(label='Unavailable', confidence=0.0, message='No audio data received.')

    try:
        import io
        import librosa
        import cv2
        
        content = await audio_file.read()
        
        y, sr = librosa.load(io.BytesIO(content), sr=22050, duration=5.0, mono=True)
        target_len = int(22050 * 5.0)
        y = np.pad(y, (0, max(0, target_len - len(y))), mode="constant")[:target_len]
        
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=2000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_n = ((mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8) * 255).astype(np.uint8)
        mel_r = cv2.resize(mel_n, (224, 224), interpolation=cv2.INTER_LINEAR)
        mel_rgb = np.stack([mel_r] * 3, axis=-1).astype(np.float32) / 255.0
        audio_array = np.expand_dims(mel_rgb, axis=0)
        
        model = load_model(settings.audio_model_path)
        if model is None:
            return BranchScore(label='Error', confidence=0.0, message='Audio model unavailable.')
        
        prediction = model.predict(audio_array, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction) * 100)
        
        classes = ["asthma", "copd", "healthy", "pneumonia"]
        label = classes[class_idx].capitalize()
        
        return BranchScore(label=label, confidence=confidence, message=f'Respiratory sounds: {label} ({confidence:.1f}%)')
    except Exception as e:
        return BranchScore(label='Error', confidence=0.0, message=f'Audio processing failed: {str(e)}')

async def evaluate_tabular_branch(tabular_data: dict[str, Any], settings: Settings) -> BranchScore:
    if not tabular_data:
        return BranchScore(label='Unavailable', confidence=0.0, message='No clinical profile data received.')

    try:
        feature_vector = np.array([
            tabular_data.get('age', 0),
            1.0 if tabular_data.get('gender') == 'M' else 0.0,
            1.0 if tabular_data.get('smoking_status') == 'current' else 0.0,
            tabular_data.get('fev1_percent', 80),
            tabular_data.get('spo2', 95),
            tabular_data.get('respiratory_rate', 16),
            tabular_data.get('cough_severity', 0),
            1.0 if tabular_data.get('wheeze') == 'yes' else 0.0,
            1.0 if tabular_data.get('chest_tightness') == 'yes' else 0.0,
            1.0 if tabular_data.get('crackles') == 'yes' else 0.0,
            1.0 if tabular_data.get('fever') == 'yes' else 0.0,
            tabular_data.get('bmi', 25),
            tabular_data.get('copd_exacerbations', 0),
        ], dtype=np.float32)
        
        scaler = get_tabular_scaler()
        if scaler is not None:
            tabular_array = scaler.transform([feature_vector]).astype(np.float32)
        else:
            tabular_array = np.expand_dims(feature_vector, axis=0)
        
        model = load_model(settings.tabular_model_path)
        if model is None:
            return BranchScore(label='Error', confidence=0.0, message='Tabular model unavailable.')
        
        prediction = model.predict(tabular_array, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction) * 100)
        
        classes = ["asthma", "copd", "healthy", "pneumonia"]
        label = classes[class_idx].capitalize()
        
        return BranchScore(label=label, confidence=confidence, message=f'Clinical profile: {label} ({confidence:.1f}%)')
    except Exception as e:
        return BranchScore(label='Error', confidence=0.0, message=f'Tabular processing failed: {str(e)}')

def build_support_report(image_score: BranchScore, audio_score: BranchScore, tabular_score: BranchScore) -> FinalReport:
    """Build final report combining all three branches with weighted confidence.
    
    Weighted combination: imaging 50%, tabular 35%, audio 15%
    These weights match the notebook's fusion strategy.
    """
    overall_confidence = round((
        image_score.confidence * 0.50 + 
        tabular_score.confidence * 0.35 +
        audio_score.confidence * 0.15
    ), 1)
    
    if image_score.confidence >= 60:
        primary_finding = f'{image_score.label} (imaging analysis)'
    elif tabular_score.confidence >= 60:
        primary_finding = f'{tabular_score.label} (clinical profile)'
    elif audio_score.confidence >= 60:
        primary_finding = f'{audio_score.label} (respiratory sounds)'
    else:
        primary_finding = 'Low confidence across all modalities—clinical correlation essential'
    
    recommendation = []
    
    if overall_confidence >= 70:
        recommendation.append('Strong multimodal agreement detected. Clinical evaluation strongly recommended.')
    elif overall_confidence >= 50:
        recommendation.append('Moderate multimodal evidence. Further clinical assessment advised.')
    else:
        recommendation.append('Low multimodal confidence. Continue standard clinical protocols.')
    
    if image_score.label.lower() != 'healthy':
        recommendation.append(f'Imaging findings suggest {image_score.label}—review with radiology.')
    
    if tabular_score.label.lower() != 'healthy':
        recommendation.append(f'Clinical profile consistent with {tabular_score.label}—monitor vital trends.')
    
    if audio_score.label.lower() != 'healthy':
        recommendation.append(f'Respiratory sounds suggest {audio_score.label}—auscultation correlation needed.')
    
    recommendation.append('All results are supportive decision aids only—not diagnostic.')
    
    note = (
        f'Multimodal Assessment (Imaging 50% | Clinical Profile 35% | Respiratory Sounds 15%) | '
        f'Overall confidence: {overall_confidence}% | Physician review required'
    )
    
    return FinalReport(
        primary_finding=primary_finding,
        overall_confidence=overall_confidence,
        recommendation=recommendation,
        note=note,
    )

