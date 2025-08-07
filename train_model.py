import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from imblearn.combine import SMOTETomek

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

from utils import clean_payload
from lstm import build_lstm_model

# === Konfigurasi ===
DATA_PATH = 'data/XSS_dataset.csv'
MODEL_PATH = 'models/xss_lstm_model.keras'
TOKENIZER_PATH = 'models/tokenizer.pkl'
MAX_LEN = 200
EPOCHS = 10
BATCH_SIZE = 32
N_SPLITS = 5

# === Load dan Preprocessing ===
df = pd.read_csv(DATA_PATH)
df['Payload'] = df['Payload'].astype(str).apply(clean_payload)
df['Label'] = df['Label'].astype(int)
df = df[df['Label'].isin([0, 1])]

print("Distribusi Label:\n", df['Label'].value_counts())

# Tokenisasi & Padding
tokenizer = Tokenizer(char_level=False)
tokenizer.fit_on_texts(df['Payload'])
sequences = tokenizer.texts_to_sequences(df['Payload'])
X = pad_sequences(sequences, maxlen=MAX_LEN)
y = df['Label'].values
vocab_size = len(tokenizer.word_index) + 1

# Resampling
smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X, y)

# Simpan tokenizer
os.makedirs('models', exist_ok=True)
joblib.dump(tokenizer, TOKENIZER_PATH)

# === Fungsi Evaluasi ===
def evaluate_fold(model, X_test, y_test, threshold=0.5):
    y_probs = model.predict(X_test).flatten()
    y_pred = (y_probs > threshold).astype(int)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

# === Cross-Validation ===
kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
fold_results = []

for i, (train_idx, test_idx) in enumerate(kfold.split(X_resampled, y_resampled)):
    print(f"\n📂 Fold {i+1}")
    X_train, X_test = X_resampled[train_idx], X_resampled[test_idx]
    y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]

    model = build_lstm_model(vocab_size, MAX_LEN)
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],
        verbose=0
    )

    metrics = evaluate_fold(model, X_test, y_test, threshold=0.5)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    fold_results.append(metrics)

# === Rata-Rata Cross-Validation ===
avg_metrics = pd.DataFrame(fold_results).mean()
print("\n📊 Rata-rata dari 5 Fold:")
print(avg_metrics)

# === Training Final Model dengan Seluruh Data ===
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
model = build_lstm_model(vocab_size, MAX_LEN)
history = model.fit(
    X_train_full, y_train_full,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
    verbose=1
)

# Simpan model
model.save(MODEL_PATH)

# === Evaluasi Final ===
train_loss, train_acc = model.evaluate(X_train_full, y_train_full, verbose=0)
test_loss, test_acc = model.evaluate(X_test_full, y_test_full, verbose=0)
print(f"\n✅ Final Train Accuracy: {train_acc:.4f}")
print(f"✅ Final Test Accuracy : {test_acc:.4f}")

# Laporan Akhir
X_eval = pad_sequences(tokenizer.texts_to_sequences(df['Payload']), maxlen=MAX_LEN)
y_true = df['Label'].values
y_probs = model.predict(X_eval).flatten()
y_pred = (y_probs > 0.3).astype(int)

print("\n=== Evaluation Report on Cleaned Dataset ===")
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=['Benign', 'Malicious']))

# === Visualisasi Training Final & Confusion Matrix ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='x')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='x')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrix(y_true, y_pred), cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()

plt.xticks(ticks=[0, 1], labels=['Benign', 'Malicious'])
plt.yticks(ticks=[0, 1], labels=['Benign', 'Malicious'])
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.show()
