import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Dynamic directory resolution relative to this file
APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent

class Settings(BaseSettings):
    # API Server Settings
    APP_TITLE: str = "E-commerce Text Classification API"
    APP_VERSION: str = "1.0.0"
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Model Files Config
    MODEL_DIR: Path = APP_DIR / "model"
    MODEL_PATH: Path = APP_DIR / "model" / "model.pth"
    VECTORIZER_PATH: Path = APP_DIR / "model" / "vectorizer.pkl"
    LABEL_ENCODER_PATH: Path = APP_DIR / "model" / "label_encoder.pkl"

    # Environment overrides
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
