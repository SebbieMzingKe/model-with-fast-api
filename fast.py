import io
import pickle
import tensorflow as tf

import numpy as np
import PIL.Image
import PIL.ImageOps
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# with open("trained_plant_disease_model1.keras", "rb") as m:
model = tf.keras.models.load_model("trained_plant_disease_model1.keras")

class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 
    'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)
    


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    print(file)
    contents = await file.read()
    # print(contents)
    pil_image = PIL.Image.open(io.BytesIO(contents)).convert("RGB")
    pil_image = PIL.ImageOps.invert(pil_image)
    pil_image = pil_image.resize((128, 128), PIL.Image.Resampling.LANCZOS)
    img_array = np.array(pil_image).reshape(1, 128, 128, 3)
    prediction = model.predict(img_array)
    # print("Prediction",prediction)
    print("Prediction: ", int(np.argmax(prediction[0])))
    predicted_class = int(np.argmax(prediction))
    predicted_disease = class_labels[predicted_class]
    return {"Prediction": predicted_disease}