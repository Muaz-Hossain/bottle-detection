from fastapi import FastAPI,Path,Query,File,UploadFile
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()
model = YOLO('best.pt')

@app.post('/detect')
async def detect_image(file : UploadFile=File(...)):
    contents = await file.read()

    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    results = model(img)[0]

    detections = []
    for box in results.boxes:
        detections.append({
            "class": model.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 2),
            "box": [int(v) for v in box.xyxy[0].tolist()],
        })

    return {"count": len(detections), "detections": detections}


