# client/streamlit_app.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import streamlit as st
from PIL import Image

from app.core.config import settings
from app.models.load import (
    age_gender_model, emotion_model,
    AGE_BINS, GENDERS, EMOTIONS, AGE_BIN_CENTERS
)
from app.vision.detect import detect_faces, crop_and_resize, draw_overlays

st.set_page_config(page_title="AgeEstAI", page_icon="👤", layout="wide")

st.title("👤 AgeEstAI — Real-time Age, Gender & Emotion")
st.caption("Allow camera → live predictions overlayed on your video feed. (Non-diagnostic)")

# ── Status row (no one-line conditionals) ──────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.success("Age/Gender model loaded ✅")
with c2:
    if emotion_model is not None:
        st.success("Emotion model loaded ✅")
    else:
        st.warning("Emotion model not found — showing '—' for Emotion")

# ── Layout: left = camera/output (on top), right = settings ───────────────────
left, right = st.columns([2, 1], gap="large")

with right:
    st.subheader("Settings")
    conf = st.slider("Face detection confidence", 0.3, 0.9, 0.60, 0.05)
    max_faces = st.slider("Max faces", 1, 8, settings.MAX_FACES, 1)
    temp = st.slider("Age temperature (sharper ⟵ 0.6…1.4 ⟶ smoother)", 0.6, 1.4, settings.AGE_SOFTMAX_T, 0.05)

    st.markdown("---")
    st.write("**Models**")
    st.text(f"Age/Gender: {settings.AGE_GENDER_MODEL_PATH}")
    st.text(f"Emotion:    {settings.EMOTION_MODEL_PATH if settings.EMOTION_MODEL_PATH else '(auto)'}")

with left:
    # Camera FIRST so it stays high in the layout
    camera = st.camera_input("Camera", key="camera", label_visibility="collapsed")

    if camera is not None:
        t0 = time.time()

        img = Image.open(camera).convert("RGB")
        rgb = np.array(img)

        # Detect faces
        boxes = detect_faces(rgb, min_score=conf)[:max_faces]

        # Crop + resize to model size (224x224)
        faces = crop_and_resize(rgb, boxes, out_size=settings.IMG_SIZE)

        if len(faces) > 0:
            # ── Age/Gender
            age_logits, gen_logits = age_gender_model.predict(faces, verbose=0)

            # Temperature scaling for age & expected years
            age_probs = np.exp(np.log(np.clip(age_logits, 1e-8, 1.0)) / temp)
            age_probs = age_probs / age_probs.sum(axis=1, keepdims=True)
            age_bins_idx = age_probs.argmax(axis=-1)
            ages_bin = [AGE_BINS[i] for i in age_bins_idx.tolist()]
            age_years = (age_probs @ np.array(AGE_BIN_CENTERS)).tolist()

            genders = [GENDERS[i] for i in gen_logits.argmax(-1).tolist()]

            # ── Emotion
            if emotion_model is not None:
                emo_logits = emotion_model.predict(faces, verbose=0)
                emos = [EMOTIONS[i] for i in emo_logits.argmax(-1).tolist()]
            else:
                emos = ["—"] * len(faces)

            # Compose labels e.g. "Age 26.8 (20–29) | male | happy"
            labels = [
                f"Age {ay:.1f} ({ab}) | {g} | {e}"
                for ay, ab, g, e in zip(age_years, ages_bin, genders, emos)
            ]
            out = draw_overlays(rgb, boxes, labels)
        else:
            out = rgb

        fps = 1.0 / max(1e-6, (time.time() - t0))
        st.caption(f"FPS: {fps:.2f}")
        # Use legacy arg for older Streamlit if needed
        try:
            st.image(out, channels="RGB", use_container_width=True)
        except TypeError:
            st.image(out, channels="RGB", use_column_width=True)
    else:
        st.info("Click **Allow** to enable the camera and start predictions.")
