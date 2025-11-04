import streamlit as st, cv2, requests, numpy as np

st.set_page_config(page_title="AgeEstAI", layout="wide")
st.title("AgeEstAI – Real-time Age/Gender/Emotion")
endpoint = st.text_input("API endpoint", "http://localhost:8000/infer")
run = st.toggle("Start camera", value=False)

cap = None
if run:
    cap = cv2.VideoCapture(0)

ph = st.empty()
while run:
    ok, frame = cap.read()
    if not ok:
        st.write("No camera access"); break
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        continue

    files = {"file": ("frame.jpg", buf.tobytes(), "image/jpeg")}
    try:
        r = requests.post(endpoint, files=files, timeout=5)
        res = r.json()
        for (x1,y1,x2,y2,_), age, gen, emo in zip(res["boxes"], res["ages"], res["genders"], res["emotions"]):
            label = f"{age} | {'M' if gen==1 else 'F'} | {emo}"
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,label,(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.putText(frame,f"FPS:{res.get('fps',0.0):.1f}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
    except Exception:
        cv2.putText(frame,"API not reachable",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

    ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    run = st.session_state.get("Start camera", True)

if cap:
    cap.release()
