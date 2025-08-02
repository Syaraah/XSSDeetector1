import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from imblearn.combine import SMOTETomek 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

import joblib
from preprocessing.text_cleaning import clean_payload

DATA_PATH = 'XSS_dataset.csv'
MODEL_PATH = 'models/xss_lstm_model.keras'
TOKENIZER_PATH = 'models/tokenizer.pkl'
MAX_LEN = 200

# Load dan preprocessing data
df = pd.read_csv(DATA_PATH)
df['Payload'] = df['Payload'].astype(str).apply(clean_payload)
df['Label'] = df['Label'].astype(int)
df = df[df['Label'].isin([0, 1])]  

print("Distribusi Label:")
print(df['Label'].value_counts())

# Tokenisasi & Padding 
tokenizer = Tokenizer(char_level=False)
tokenizer.fit_on_texts(df['Payload'])
sequences = tokenizer.texts_to_sequences(df['Payload']) 
X = pad_sequences(sequences, maxlen=MAX_LEN)
y = df['Label'].values
vocab_size = len(tokenizer.word_index) + 1

# Resampling
smt = SMOTETomek(random_state=42)
x_resampled, y_resampled = smt.fit_resample(X, y)

# Simpan tokenizer
os.makedirs('models', exist_ok=True)
joblib.dump(tokenizer, TOKENIZER_PATH)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

# LSTM Model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=MAX_LEN))
model.add(LSTM(64, return_sequences=True))  
model.add(Dropout(0.3))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))

model.compile(loss='binary_crossentropy', optimizer=Adam(1e-3), metrics=['accuracy'])

# Training
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Evaluasi Train/Test
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"\n✅ Train Accuracy: {train_acc:.4f}")
print(f"✅ Test Accuracy : {test_acc:.4f}")

print("\n=== Evaluation Report ===")
X_all = pad_sequences(tokenizer.texts_to_sequences(df['Payload']), maxlen=MAX_LEN)
y_true = df['Label'].values
y_probs = model.predict(X_all).flatten()
y_pred = (y_probs > 0.4).astype(int)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=['Benign', 'Malicious']))  

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malicious'])

plt.figure(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

model.save(MODEL_PATH)

# Plot hasil training 
plt.figure(figsize=(12, 5))

# Plot akurasi
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='x')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='x')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
