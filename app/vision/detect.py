# app/vision/detect.py
from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple
from app.core.config import settings

# ---- FaceDetector using OpenCV Haar cascades (no extra downloads) ----
class FaceDetector:
    def __init__(self, min_neighbors: int = 5, scale_factor: float = 1.1):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.det = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, bgr: np.ndarray, min_size: Tuple[int,int] = (64,64)) -> List[List[int]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rects = self.det.detectMultiScale(
            gray, scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors, minSize=min_size
        )
        # convert to [x1,y1,x2,y2]
        boxes = [[int(x), int(y), int(x+w), int(y+h)] for (x,y,w,h) in rects]
        # prioritize larger faces
        boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        return boxes[: settings.MAX_FACES]

# ---- Streamlit helpers (RGB space) ----
def detect_faces(image_rgb: np.ndarray, min_score: float = 0.6):
    """Compatibility wrapper: run Haar (no scores), return boxes."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    det = FaceDetector()
    return det.detect(bgr)

def crop_and_resize(image_rgb: np.ndarray, boxes, out_size: int | None = None):
    """Batch crop+resize for Streamlit pipeline."""
    from app.models.preprocess import crop_and_resize as _single
    faces = []
    if out_size is None:
        out_size = settings.IMG_SIZE
    for b in boxes:
        face = _single(image_rgb, b, out_size)
        if face is not None:
            faces.append(face)
    if not faces:
        return np.zeros((0, out_size, out_size, 3), dtype=np.float32)
    return np.stack(faces, axis=0)

def draw_overlays(image_rgb: np.ndarray, boxes: List[List[int]], labels: list[str]):
    out = image_rgb.copy()
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    for (x1,y1,x2,y2), text in zip(boxes, labels):
        cv2.rectangle(bgr, (x1,y1), (x2,y2), (0,255,0), 2)
        # text background
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y0 = max(0, y1 - th - 6)
        cv2.rectangle(bgr, (x1, y0), (x1 + tw + 6, y0 + th + 6), (0,0,0), -1)
        cv2.putText(bgr, text, (x1+3, y0+th+1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
