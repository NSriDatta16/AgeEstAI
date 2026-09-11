from __future__ import annotations

import cv2
import numpy as np


def crop_and_resize(image_rgb: np.ndarray, box, out_size: int = 224):
    """Crop [x1,y1,x2,y2] from an RGB image and return an RGB float image."""
    if image_rgb is None or image_rgb.size == 0:
        return None

    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = image_rgb.shape[:2]
    x1, x2 = max(0, min(x1, w - 1)), max(1, min(x2, w))
    y1, y2 = max(0, min(y1, h - 1)), max(1, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image_rgb[y1:y2, x1:x2]
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return crop.astype(np.float32) / 255.0
