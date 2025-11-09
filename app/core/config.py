# app/core/config.py
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App
    APP_NAME: str = "AgeEstAI"

    # Image / detection
    IMG_SIZE: int = 224
    MAX_FACES: int = 5

    # Models
    AGE_GENDER_MODEL_PATH: str = "models/age_gender_finetuned.keras"
    EMOTION_MODEL_PATH: str = "models/emotion_finetuned.keras"

    # Calibration (UI slider default)
    AGE_SOFTMAX_T: float = 0.9

    # Pydantic v2 settings
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

settings = Settings()
