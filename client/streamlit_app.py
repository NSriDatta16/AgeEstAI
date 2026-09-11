import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import streamlit as st
from PIL import Image

from app.core.config import settings
from app.models.infer import infer_image
from app.vision.detect import draw_overlays

st.set_page_config(page_title="AgeEstAI", page_icon="👤", layout="wide")

st.title("👤 AgeEstAI — Age, Gender & Emotion")
st.caption("Pretrained facial-attribute models with explicit age-bin mapping. Non-diagnostic.")

left, right = st.columns([2, 1], gap="large")

with right:
    st.subheader("Settings")
    max_faces = st.slider("Max faces", 1, 8, settings.MAX_FACES, 1)
    st.markdown("---")
    st.write("**Age bins**")
    st.code("0–12\n13–19\n20–29\n30–39\n40–49\n50–64\n65+")
    st.info("Age is estimated continuously first, then mapped deterministically into the displayed bin.")

with left:
    camera = st.camera_input("Camera", key="camera")

    if camera is not None:
        t0 = time.perf_counter()
        img = Image.open(camera).convert("RGB")
        rgb = np.array(img)
        bgr = rgb[:, :, ::-1].copy()

        try:
            predictions = infer_image(bgr, max_faces=max_faces)
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            predictions = []

        boxes = []
        labels = []
        for p in predictions:
            area = p.get("facial_area", {})
            if not {"x", "y", "w", "h"}.issubset(area):
                continue
            x, y, w, h = int(area["x"]), int(area["y"]), int(area["w"]), int(area["h"])
            boxes.append([x, y, x + w, y + h])
            labels.append(
                f"Age {p['age']:.1f} ({p['age_bin']}) | {p['gender']} | "
                f"{p['emotion']} ({p['emotion_confidence']:.0f}%)"
            )

        out = draw_overlays(rgb, boxes, labels) if boxes else rgb
        fps = 1.0 / max(1e-6, time.perf_counter() - t0)
        st.caption(f"Faces: {len(predictions)} · Inference FPS: {fps:.2f}")
        st.image(out, channels="RGB", use_container_width=True)

        if predictions:
            st.subheader("Predictions")
            for idx, p in enumerate(predictions, 1):
                st.write(
                    f"**Face {idx}:** age **{p['age']:.1f}** → **{p['age_bin']}**, "
                    f"gender **{p['gender']}**, emotion **{p['emotion']}** "
                    f"({p['emotion_confidence']:.1f}%)"
                )
        else:
            st.warning("No face was detected. Try better lighting and a frontal view.")
    else:
        st.info("Click **Allow** to enable the camera and start predictions.")
