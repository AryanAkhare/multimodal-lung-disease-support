from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Backend configuration: model paths and API settings."""
    base_path: Path = Path(__file__).parent.parent
    
    # Model paths (all .keras format)
    image_model_path: str = str(Path(__file__).parent.parent / 'artifacts' / 'image_binary_model.keras')
    audio_model_path: str = str(Path(__file__).parent.parent / 'artifacts' / 'audio_binary_model.keras')
    tabular_model_path: str = str(Path(__file__).parent.parent / 'artifacts' / 'tabular_profile_model.keras')
    global_model_path: str = str(Path(__file__).parent.parent / 'artifacts' / 'best_global_model.keras')
    tabular_scaler_path: str = str(Path(__file__).parent.parent / 'artifacts' / 'tabular_scaler.joblib')
    
    # API settings
    api_title: str = 'Multimodal Lung Disease Support API'
    debug: bool = True
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
