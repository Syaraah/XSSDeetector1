import os
import json
import tensorflow as tf
import joblib
import numpy as np

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from tensorflow.keras.preprocessing.sequence import pad_sequences


from utils import clean_payload

# === KONFIGURASI ===
MODEL_PATH = 'models/xss_lstm_model.keras'
TOKENIZER_PATH = 'models/tokenizer.pkl'
MAX_LEN = 200
THRESHOLD = 0.3

# === INISIALISASI ===
app = FastAPI(
    title="XSS Detection UI",
    description="UI Web untuk mendeteksi serangan XSS.",
    version="2.0"
)

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# === LOAD MODEL DAN TOKENIZER ===
model = tf.keras.models.load_model(MODEL_PATH)
tokenizer = joblib.load(TOKENIZER_PATH)

LOG_FILE = "logs/detection_logs.json"

# === Halaman utama: input form ===
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

# === Menangani submit form POST ===
@app.post("/", response_class=HTMLResponse)
async def submit_form(request: Request, payload: str = Form(...)):
    # Dummy prediksi (ganti dengan modelmu)
    prediction = "XSS Detected" if "<script>" in payload else "Clean"

    # Simpan ke log JSON
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "payload": payload,
        "prediction": prediction
    }

    # Pastikan direktori log ada
    os.makedirs("logs", exist_ok=True)

    # Load file log jika ada
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

    return templates.TemplateResponse("form.html", {
        "request": request,
        "prediction": prediction,
        "payload": payload
    })
