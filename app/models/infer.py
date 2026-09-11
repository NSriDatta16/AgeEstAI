from __future__ import annotations

from typing import Any

import numpy as np
from deepface import DeepFace

AGE_BINS = ["0-12", "13-19", "20-29", "30-39", "40-49", "50-64", "65+"]
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
GENDERS = ["female", "male"]


def age_to_bin(age: float) -> str:
    """Convert the continuous apparent-age estimate into the project's bins."""
    age = float(age)
    if age <= 12:
        return "0-12"
    if age <= 19:
        return "13-19"
    if age <= 29:
        return "20-29"
    if age <= 39:
        return "30-39"
    if age <= 49:
        return "40-49"
    if age <= 64:
        return "50-64"
    return "65+"


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    age = _as_float(result.get("age", 0.0))
    emotion_scores = result.get("emotion") or {}
    gender_scores = result.get("gender") or {}

    emotion = str(result.get("dominant_emotion", "neutral")).lower()
    if emotion not in EMOTIONS:
        emotion = "neutral"

    dominant_gender = str(result.get("dominant_gender", "Unknown")).lower()
    gender = "female" if dominant_gender in {"woman", "female"} else "male" if dominant_gender in {"man", "male"} else "unknown"

    return {
        "age": round(age, 1),
        "age_bin": age_to_bin(age),
        "gender": gender,
        "emotion": emotion,
        "emotion_confidence": round(_as_float(emotion_scores.get(emotion, 0.0)), 2),
        "gender_confidence": round(_as_float(gender_scores.get("Woman" if gender == "female" else "Man", 0.0)), 2),
        "face_confidence": round(_as_float(result.get("face_confidence", result.get("confidence", 0.0))), 3),
        "facial_area": result.get("region") or result.get("facial_area") or {},
    }


def infer_image(image_bgr: np.ndarray, max_faces: int = 5) -> list[dict[str, Any]]:
    """Analyze all detected faces in one image.

    DeepFace performs face detection, alignment and the age/emotion inference using
    pretrained facial-attribute models. This replaces the previous custom classifier
    path, which depended on missing local model artifacts and inconsistent labels.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    results = DeepFace.analyze(
        img_path=image_bgr,
        actions=["age", "gender", "emotion"],
        detector_backend="opencv",
        enforce_detection=False,
        align=True,
        silent=True,
        anti_spoofing=False,
    )

    if isinstance(results, dict):
        results = [results]

    normalized = [_normalize_result(item) for item in results]
    normalized.sort(
        key=lambda item: (
            item["facial_area"].get("w", 0) * item["facial_area"].get("h", 0)
        ),
        reverse=True,
    )
    return normalized[:max_faces]


def infer_batch(faces: list[np.ndarray]):
    """Compatibility helper for callers that already provide face crops."""
    ages: list[float] = []
    genders: list[str] = []
    emotions: list[str] = []

    for face in faces:
        results = DeepFace.analyze(
            img_path=face,
            actions=["age", "gender", "emotion"],
            detector_backend="skip",
            enforce_detection=False,
            align=True,
            silent=True,
            anti_spoofing=False,
        )
        if isinstance(results, dict):
            results = [results]
        item = _normalize_result(results[0]) if results else {
            "age": 0.0,
            "gender": "unknown",
            "emotion": "neutral",
        }
        ages.append(item["age"])
        genders.append(item["gender"])
        emotions.append(item["emotion"])

    return ages, genders, emotions
