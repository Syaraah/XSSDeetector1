from flask import Flask, render_template, request
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import clean_payload
import os

# === Konfigurasi ===
MODEL_PATH = 'models/xss_lstm_model.keras'
TOKENIZER_PATH = 'models/tokenizer.pkl'
MAX_LEN = 200
THRESHOLD = 0.3

# === Inisialisasi Flask ===
app = Flask(__name__)

# === Load model dan tokenizer ===
model = tf.keras.models.load_model(MODEL_PATH)
tokenizer = joblib.load(TOKENIZER_PATH)

# === Route Utama ===
@app.route('/')
def index():
    return render_template('index.html')

# === Route Prediksi ===
@app.route('/predict', methods=['POST'])
def predict():
    payload = request.form['payload']

    if not payload.strip():
        return render_template('index.html', error="Masukkan payload terlebih dahulu.")

    cleaned = clean_payload(payload)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)

    prob = model.predict(padded)[0][0]
    label = "Malicious" if prob > THRESHOLD else "Benign"

    return render_template(
        'result.html',
        original=payload,
        cleaned=cleaned,
        label=label,
        confidence=f"{prob:.2f}"
    )

# === Jalankan Flask App ===
if __name__ == '__main__':
    app.run(debug=True)
