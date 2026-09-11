from __future__ import annotations

import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
def home():
    return """<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>AgeEstAI</title>
<style>body{font-family:system-ui;max-width:900px;margin:40px auto;padding:0 20px}button{padding:10px 16px;margin-top:12px}img{max-width:100%;margin-top:20px;border-radius:12px}pre{background:#f4f4f4;padding:16px;border-radius:10px;overflow:auto}</style></head>
<body><h1>AgeEstAI</h1><p>Corrected pretrained age, age-bin and emotion inference.</p>
<input id='file' type='file' accept='image/*'><br><button onclick='run()'>Analyze</button>
<div id='status'></div><img id='preview'><pre id='out'></pre>
<script>
async function run(){const f=document.getElementById('file').files[0];if(!f)return;
const fd=new FormData();fd.append('file',f);document.getElementById('status').textContent='Analyzing...';
const r=await fetch('/infer',{method:'POST',body:fd});const j=await r.json();
document.getElementById('status').textContent=r.ok?'Done':'Error';document.getElementById('out').textContent=JSON.stringify(j,null,2);}
</script></body></html>"""


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
