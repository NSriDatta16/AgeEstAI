# AgeEstAI
Real-time **Age, Gender & Emotion** detection (TensorFlow + FastAPI + MediaPipe + Streamlit).

## Run (local)
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.api.main:app --reload
# new terminal
streamlit run client/streamlit_app.py
