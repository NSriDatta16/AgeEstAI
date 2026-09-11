from app.models.infer import AGE_BINS, EMOTIONS, GENDERS, age_to_bin, infer_batch, infer_image

AGE_BIN_CENTERS = [6, 16, 25, 35, 45, 57, 70]

# Backward-compatible names for the previous Streamlit client.
age_gender_model = None
emotion_model = None

__all__ = [
    "AGE_BINS",
    "EMOTIONS",
    "GENDERS",
    "AGE_BIN_CENTERS",
    "age_to_bin",
    "infer_batch",
    "infer_image",
    "age_gender_model",
    "emotion_model",
]
