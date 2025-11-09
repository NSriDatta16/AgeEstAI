# 👤 AgeEstAI
**Real-time Age, Gender & Emotion Estimation using Deep Learning + FastAPI + Streamlit**

A lightweight, end-to-end computer vision system that estimates **age (in years and age group)**, **gender**, and **emotion** from live webcam input — powered by **TensorFlow/Keras**, **FastAPI**, and **Streamlit**, with a **Dockerized deployment** pipeline.

---

## 🚀 Tech Stack

| Layer           | Tools & Frameworks |
|-----------------|--------------------|
| **Modeling**    | TensorFlow / Keras (CNN, multitask learning), transfer learning (MobileNetV3) |
| **Vision**      | MediaPipe Face Detection (cropping, alignment, normalization 224×224) |
| **Backend**     | FastAPI (`/infer` returns JSON predictions) |
| **Frontend**    | Streamlit (real-time webcam overlay, FPS) |
| **Container**   | Docker (single image for UI or API) |

---

## 🧠 Features
- 🔍 **Face detection** → automatic cropping & normalization  
- 🧒 **AgeGenderNet** → predicts **age bin (7 classes)** + **gender (2 classes)** and shows **expected age in years**  
- 😊 **EmotionNet** → classifies **7 emotions** (*angry, disgust, fear, happy, sad, surprise, neutral*)  
- 🎥 **Streamlit UI** → live camera with overlays & FPS  
- ⚙️ **FastAPI** → `/infer` accepts an image and returns JSON  
- 📦 **Docker-ready** → run locally or deploy on cloud

---

## 📂 Project Layout

```text
AgeEstAI/
├─ app/
│  ├─ api/                # FastAPI backend
│  │  └─ main.py
│  ├─ core/               # config, settings
│  │  └─ config.py
│  ├─ models/             # model loading & inference utils
│  │  ├─ load.py
│  │  └─ infer.py
│  └─ vision/             # face detection utils
│     └─ detect.py
├─ client/
│  └─ streamlit_app.py    # real-time Streamlit UI
├─ models/
│  ├─ age_gender_finetuned.keras
│  └─ emotion_finetuned.keras
├─ training/
│  ├─ 2_train_multitask_tf.py
│  └─ 2_train_emotion_only.py
├─ Dockerfile
├─ .dockerignore
├─ requirements.txt
└─ README.md
