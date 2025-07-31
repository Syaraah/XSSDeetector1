import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import joblib

from preprocessing.text_cleaning import clean_payload

# ==== 1. Load & Preprocess Data ====
DATA_PATH = 'XSS_dataset.csv'
MODEL_PATH = 'models/xss_lstm_model.h5'
TOKENIZER_PATH = 'models/tokenizer.pkl'

df = pd.read_csv(DATA_PATH)
df['Payload'] = df['Payload'].astype(str).apply(clean_payload)
df['Label'] = df['Label'].astype(int)
df = df[df['Label'].isin([0, 1])]  # hanya label 0 dan 1

# Cek distribusi awal
print("Distribusi awal Label:")
print(df['Label'].value_counts())

# ==== 2. Balancing Dataset ====
df_major = df[df['Label'] == 1]  # XSS
df_minor = df[df['Label'] == 0]  # benign

df_minor_upsampled = resample(df_minor, replace=True, n_samples=len(df_major), random_state=42)
df = pd.concat([df_major, df_minor_upsampled]).sample(frac=1).reset_index(drop=True)

print("\nSetelah balancing:")
print(df['Label'].value_counts())

# ==== 3. Tokenisasi & Padding ====
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['Payload'])

sequences = tokenizer.texts_to_sequences(df['Payload'])
max_len = 200  # panjang input

X = pad_sequences(sequences, maxlen=max_len, padding='post')
y = df['Label'].values

# Simpan tokenizer
os.makedirs('models', exist_ok=True)
joblib.dump(tokenizer, TOKENIZER_PATH)

# ==== 4. Train/Test Split ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== 5. Bangun Model LSTM ====
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_len),
    LSTM(64),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=Adam(1e-3), metrics=['accuracy'])

# ==== 6. Training ====
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# ==== 7. Evaluasi ====
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"\n✅ Train Accuracy: {train_acc:.4f}")
print(f"✅ Test Accuracy : {test_acc:.4f}")

# ==== 8. Simpan Model ====
model.save(MODEL_PATH)
print(f"\n📦 Model saved to: {MODEL_PATH}")

# ==== 9. Uji Coba Payload ====
examples = [
    "<script>alert('xss')</script>",
    "hello world",
    "username=admin",
    "document.cookie",
    "GET /page?search=apple"
]

print("\n🔍 Contoh Prediksi:")
for p in examples:
    clean = clean_payload(p)
    seq = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded, verbose=0)[0][0]
    label = 'XSS' if pred > 0.8 else 'Benign'
    print(f"- {p} => Prediksi: {pred:.4f} => {label}")
