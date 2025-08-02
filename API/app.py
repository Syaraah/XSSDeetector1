from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import joblib 
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing.text_cleaning import clean_payload

#Initialize FastAPI app
app = FastAPI()

model = load_model('models/xss_detection_model.h5')
tokenizer = joblib.load('models/tokenizer.pkl')
max_len = 200
THRESHOLD = 0.8
templates = Jinja2Templates(directory = "api/templates")

class PayloadInput(BaseModel):
    payload: str
    
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict_xss(input: PayloadInput):
    # Step 1: Terima input
    print("[INFO] Payload diterima:", input.payload)

    # Step 2: Bersihkan input
    cleaned = clean_payload(input.payload)
    print("[INFO] Payload dibersihkan:", cleaned)

    # Step 3: Ubah ke sequence dan padding
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')

    # Step 4: Prediksi
    pred = model.predict(padded)[0][0]
    print("[INFO] Prediksi probabilitas:", pred)

    # Step 5: Kembalikan hasil
    result = {
        "payload": input.payload,
        "cleaned_payload": cleaned,
        "prediction": "malicious" if pred > THRESHOLD else "benign",
        "confidence_score": round(float(pred), 4)
    }
    print("[INFO] Hasil:", result)
    return result

