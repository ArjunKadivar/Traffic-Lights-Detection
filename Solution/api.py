from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO

app = FastAPI()
model = YOLO('best.pt')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        return JSONResponse(
            status_code=400,
            content={"error": "File must be an image"}
        )
    
    try:
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image file"}
            )
        
        results = model.predict(source=img, conf=0.25)
        
        predictions = []
        for r in results[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = r
            predictions.append({
                "x1": round(float(x1), 1),
                "y1": round(float(y1), 1),
                "x2": round(float(x2), 1),
                "y2": round(float(y2), 1),
                "confidence": round(float(confidence), 2),
                "class_id": int(class_id),
                "class_name": model.names[int(class_id)]
            })
        
        return JSONResponse(
            status_code=200,
            content={"predictions": predictions}
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )
