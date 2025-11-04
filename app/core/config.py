from pydantic import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "AgeEstAI"
    DETECTOR: str = "mediapipe"  # or "opencv"
    IMG_SIZE: int = 224
    EMA_ALPHA: float = 0.6
    MAX_FACES: int = 5
    USE_ONNX: bool = False
    AGE_MODEL_PATH: str = "models/age_model.keras"
    GENDER_MODEL_PATH: str = "models/gender_model.keras"
    EMOTION_MODEL_PATH: str = "models/emotion_model.keras"

    class Config:
        env_file = ".env"

settings = Settings()
