from fastapi import FastAPI, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image

app = FastAPI()

model = tf.keras.models.load_model("models/lung_model.keras")

classes = ["adenocarcinoma","normal","squamous"]

@app.post("/predict")

async def predict(file: UploadFile):

    image = Image.open(file.file).resize((224,224))

    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)

    return {
        "prediction": classes[int(pred.argmax())],
        "confidence": float(pred.max())
    }