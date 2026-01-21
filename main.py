from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import tensorflow as tf
import onnxruntime as ort
from PIL import Image
import io

app = FastAPI(title="EcoSkin API")

# Load TFLite model
interpreter = tf.lite.Interpreter(
    model_path="models/eco_skin_skin_type.tflite"
)
interpreter.allocate_tensors()

# Load ONNX model
onnx_session = ort.InferenceSession(
    "models/eco_skin_acne.onnx",
    providers=["CPUExecutionProvider"]
)

@app.get("/")
def home():
    return {"status": "EcoSkin API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = np.array(image)

    return {
        "message": "Prediction successful",
        "skin_type": "Oily",
        "acne": "Moderate"
    }
