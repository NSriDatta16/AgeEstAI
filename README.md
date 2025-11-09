# 👤 AgeEstAI  
**Real-time Age, Gender & Emotion Estimation using Deep Learning + FastAPI + Streamlit**

A lightweight, end-to-end computer vision system that estimates **age (in years and age group)**, **gender**, and **emotion** from live webcam input — powered by **TensorFlow/Keras**, **FastAPI**, and **Streamlit**, with a **Dockerized deployment** pipeline.

---

## 🚀 Tech Stack
| Layer | Tools & Frameworks |
|-------|---------------------|
| **Modeling** | TensorFlow / Keras (CNN, multitask learning), transfer learning (MobileNetV3) |
| **Computer Vision** | MediaPipe Face Detection (cropping, alignment, normalization 224×224) |
| **Backend** | FastAPI (RESTful `/infer` endpoint for inference as JSON) |
| **Frontend** | Streamlit (real-time webcam overlay) |
| **Containerization** | Docker (deploy to Azure / Hugging Face Spaces / any cloud) |

---

## 🧠 Features
- 🔍 **Face detection** → automatic cropping & normalization  
- 🧒 **AgeGenderNet** → predicts **age bin (7 classes)** + **gender (2 classes)**  
- 😊 **EmotionNet** → classifies **7 emotions** (*angry, disgust, fear, happy, sad, surprise, neutral*)  
- 🎥 **Streamlit web UI** → live camera with overlays and FPS counter  
- ⚙️ **FastAPI backend** → accepts frames and returns JSON predictions  
- 📦 **Docker-ready** → single command to run locally or deploy on cloud  

---

## 📂 Project Layout
