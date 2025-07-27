import uvicorn
import numpy as np
import pickle
import requests
from fastapi import FastAPI, Request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Preprocessing.text_cleaning import normalize_payload

#Initialize FastAPI app
app = FastAPI()
model = load_model('models/xss_detection_model.h5')

with open('tokenizer/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

MAX_LEN = 200

@app.get("/")
def home():
    return {"message": "Welcome to the XSS Detection API"}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    print("[INFO] Payload received: ", data)
    raw_payload = data.get("payload", "")
    normalized = normalize_payload(raw_payload)
    seq = tokenizer.texts_to_sequences([normalized])
    padded = pad_sequences(seq, maxlen = MAX_LEN, padding = 'post')
    prediction = float(model.predict(padded)[0][0])
    return {
        "payload": raw_payload, 
        "score": prediction,
        "is_malicious": prediction > 0.5
    }
