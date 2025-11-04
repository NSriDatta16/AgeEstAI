from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import cv2, numpy as np, time

from app.vision.detect import FaceDetector
from app.models.preprocess import crop_and_resize
from app.models.infer import infer_batch

app = FastAPI(title="AgeEstAI")
detector = FaceDetector()

class InferResponse(BaseModel):
    boxes: list
    ages: list
    genders: list
    emotions: list
    fps: float

@app.post("/infer", response_model=InferResponse)
async def infer(file: UploadFile = File(...)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    t0 = time.time()
    boxes = detector.detect(img)
    faces, kept = [], []
    for b in boxes:
        face = crop_and_resize(img, b)
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
