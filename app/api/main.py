from __future__ import annotations

import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.config import settings
from app.models.infer import infer_image

app = FastAPI(title=settings.APP_NAME, version="2.0.0")


class Prediction(BaseModel):
    age: float
    age_bin: str
    gender: str
    emotion: str
    emotion_confidence: float
    gender_confidence: float
    face_confidence: float
    facial_area: dict


class InferResponse(BaseModel):
    boxes: list[list[int]]
    predictions: list[Prediction]
    fps: float


@app.post("/infer", response_model=InferResponse)
async def infer(file: UploadFile = File(...)):
    data = await file.read()
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    t0 = time.perf_counter()
    try:
        predictions = infer_image(bgr, max_faces=settings.MAX_FACES)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    boxes: list[list[int]] = []
    for item in predictions:
        area = item.get("facial_area", {})
        if {"x", "y", "w", "h"}.issubset(area):
            x, y, w, h = int(area["x"]), int(area["y"]), int(area["w"]), int(area["h"])
            boxes.append([x, y, x + w, y + h])
        else:
            boxes.append([0, 0, 0, 0])

    fps = 1.0 / max(1e-6, time.perf_counter() - t0)
    return {"boxes": boxes, "predictions": predictions, "fps": fps}


@app.get("/health")
def health():
    return {"status": "ok", "model": "DeepFace pretrained age/gender/emotion"}
