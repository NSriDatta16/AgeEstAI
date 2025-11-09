# app/api/main.py
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import cv2, numpy as np, time

from app.vision.detect import FaceDetector
from app.models.preprocess import crop_and_resize
from app.models.infer import infer_batch
from app.core.config import settings

app = FastAPI(title=settings.APP_NAME)
detector = FaceDetector()

class InferResponse(BaseModel):
    boxes: list
    ages: list[float]        # expected age (years)
    genders: list[str]
    emotions: list[str]
    fps: float

@app.post("/infer", response_model=InferResponse)
async def infer(file: UploadFile = File(...)):
    data = await file.read()
    bgr  = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return {"boxes": [], "ages": [], "genders": [], "emotions": [], "fps": 0.0}

    t0 = time.time()
    boxes = detector.detect(bgr)
    faces, kept = [], []
    for b in boxes:
        # convert BGR->RGB because our preprocess expects RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        face = crop_and_resize(rgb, b, out_size=settings.IMG_SIZE)
        if face is not None:
            faces.append(face); kept.append(b)

    if faces:
        ages, genders, emotions = infer_batch(faces)
    else:
        ages, genders, emotions = [], [], []

    fps = 1.0 / max(1e-5, (time.time()-t0))
    return {"boxes": kept, "ages": ages, "genders": genders, "emotions": emotions, "fps": fps}

@app.get("/health")
def health():
    return {"status": "ok"}
