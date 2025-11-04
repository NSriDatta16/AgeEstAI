import cv2, mediapipe as mp
mpfd = mp.solutions.face_detection

class FaceDetector:
    def __init__(self, min_conf=0.6):
        self.det = mpfd.FaceDetection(model_selection=0, min_detection_confidence=min_conf)

    def detect(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.det.process(rgb)
        faces = []
        h, w = bgr.shape[:2]
        if res.detections:
            for d in res.detections:
                box = d.location_data.relative_bounding_box
                x1 = max(0, int(box.xmin * w))
                y1 = max(0, int(box.ymin * h))
                x2 = min(w, int((box.xmin + box.width) * w))
                y2 = min(h, int((box.ymin + box.height) * h))
                score = float(d.score[0]) if d.score else 0.0
                if x2 > x1 and y2 > y1:
                    faces.append((x1, y1, x2, y2, score))
        return faces
