# AgeEstAI
Real-time **Age, Gender & Emotion** detection with a webcam.  
**Stack:** TensorFlow/Keras, FastAPI, MediaPipe, Streamlit, Docker.

## Features
- Face detection (MediaPipe) → face crop → normalized 224×224
- Two models:
  - **AgeGenderNet** → age bin (7 classes) + gender (2 classes)
  - **EmotionNet**    → 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral)
- Streamlit web UI: requests camera permission and overlays predictions
- FastAPI backend: `/infer` accepts frames and returns JSON predictions
- Docker-ready; deploy to Azure Container Apps or Hugging Face Spaces

## Project Layout
