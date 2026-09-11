# 👤 AgeEstAI

AgeEstAI is a web-deployable computer-vision application that estimates **apparent age, age group, gender, and facial emotion** from an image.

## Why the inference pipeline was changed

The previous implementation trained custom age/emotion heads but the repository did not contain the referenced model artifacts, while the UI also applied its own age-temperature logic. That made the displayed age bins dependent on the custom model's learned class boundaries and made deployment fragile.

The corrected pipeline uses **DeepFace's pretrained age, gender, and emotion models**. The model produces a continuous apparent-age estimate first. Age groups are then assigned by one deterministic mapping:

- 0–12
- 13–19
- 20–29
- 30–39
- 40–49
- 50–64
- 65+

This means the displayed bin can no longer disagree with the displayed numeric age because both come from the same age estimate.

## Architecture

```text
Browser / Streamlit
        |
        v
FastAPI /infer
        |
        v
OpenCV image decode
        |
        v
DeepFace
  ├── face detection + alignment
  ├── pretrained age model
  ├── pretrained gender model
  └── pretrained emotion model
        |
        v
Continuous age -> deterministic age bin
        |
        v
JSON predictions / browser overlay
```

DeepFace is pinned to `0.0.100`.

## Run locally

### API

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.api.main:app --reload --port 8000
```

Open `http://localhost:8000/` for the browser test page.

Health check:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

The first inference downloads/builds the pretrained DeepFace attribute models, so the first request is slower than later requests.

### Streamlit

```powershell
streamlit run client/streamlit_app.py
```

## Docker

```powershell
docker build -t ageestai:corrected .
docker run --rm -p 8000:8000 ageestai:corrected
```

Then open `http://localhost:8000/`.

## API

`POST /infer` accepts an image as multipart form data under `file`.

Example response shape:

```json
{
  "boxes": [[120, 80, 310, 290]],
  "predictions": [
    {
      "age": 27.0,
      "age_bin": "20-29",
      "gender": "male",
      "emotion": "happy",
      "emotion_confidence": 91.2,
      "gender_confidence": 97.4,
      "face_confidence": 0.998,
      "facial_area": {"x": 120, "y": 80, "w": 190, "h": 210}
    }
  ],
  "fps": 1.2
}
```

## Important model limitation

Age and emotion are **apparent facial-attribute estimates**, not ground truth. Lighting, pose, camera quality, occlusion, expression ambiguity, and demographic/model bias can materially affect results. They should not be used for high-stakes decisions.

## Project layout

```text
AgeEstAI/
├─ app/
│  ├─ api/main.py
│  ├─ core/config.py
│  ├─ models/infer.py
│  ├─ models/load.py
│  ├─ models/preprocess.py
│  └─ vision/detect.py
├─ client/streamlit_app.py
├─ training/                 # legacy dataset/custom-training experiments
├─ Dockerfile
├─ requirements.txt
└─ .github/workflows/docker.yml
```
