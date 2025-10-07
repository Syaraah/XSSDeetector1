# === Import Libraries ===
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from collections import Counter
from imblearn.combine import SMOTETomek

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

from utils import clean_payload
from lstm import build_lstm_model


# === Configuration ===
DATA_PATH = "data/XSS_dataset.csv"
TEST_DATA_PATH = "data/XSS_test_dataset.csv"
MODEL_PATH = "models/xss_lstm_model.keras"
TOKENIZER_PATH = "models/tokenizer.pkl"

MAX_LEN = 200
EPOCHS = 15
BATCH_SIZE = 64
matplotlib.use("Agg")


# === Load and preprocess dataset ===
print("📊 Loading main dataset...")
df = pd.read_csv(DATA_PATH)
df["Payload"] = df["Payload"].astype(str).apply(clean_payload)
df["Label"] = df["Label"].astype(int)
df = df[df["Label"].isin([0, 1])]

print(f"✅ Dataset loaded: {len(df)} samples")
print("Label Distribution:\n", df["Label"].value_counts())


# === Train/Val/Test Split (80-10-10) ===
print("\n📈 Splitting dataset into 80-10-10...")

X_train, X_temp, y_train, y_temp = train_test_split(
    df["Payload"], df["Label"],
    test_size=0.2,
    random_state=42,
    stratify=df["Label"]
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,   # 50% of 20% = 10% of total
    random_state=42,
    stratify=y_temp
)

print(f"\n📊 Data Split Summary:")
print(f"   Training Data: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"   Validation Data: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
print(f"   Testing Data: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")


# === Save Test Data from Split ===
test_df = pd.DataFrame({"Payload": X_test, "Label": y_test})
test_df.to_csv(TEST_DATA_PATH, index=False)
print(f"✅ Test data (split) saved to {TEST_DATA_PATH} ({len(test_df)} rows)")


# === Tokenization & Padding ===
print("\n🔤 Tokenizing and padding data...")
tokenizer = Tokenizer(char_level=False)
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_seq, maxlen=MAX_LEN)
X_val_padded = pad_sequences(X_val_seq, maxlen=MAX_LEN)
X_test_padded = pad_sequences(X_test_seq, maxlen=MAX_LEN)

y_train, y_val, y_test = np.array(y_train), np.array(y_val), np.array(y_test)


# === Resampling Training Data (SMOTE-Tomek) ===
print("\n⚖️ Applying SMOTE-Tomek resampling on training data...")
smt = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smt.fit_resample(X_train_padded, y_train)

label_map = {0: "Benign", 1: "Malicious (XSS)"}

print("\nBefore resampling:")
for lbl, count in Counter(y_train).items():
    print(f"- {label_map[lbl]}: {count:,} samples")

print("\nAfter resampling:")
for lbl, count in Counter(y_train_resampled).items():
    print(f"- {label_map[lbl]}: {count:,} samples")


# === Save Tokenizer ===
os.makedirs("models", exist_ok=True)
joblib.dump(tokenizer, TOKENIZER_PATH)
print(f"✅ Tokenizer saved to {TOKENIZER_PATH}")


# === Train Model ===
print("\n🚀 Training LSTM model...")
model = build_lstm_model(vocab_size, MAX_LEN)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train_resampled, y_train_resampled,
    validation_data=(X_val_padded, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)


# === Evaluation on Validation Data ===
print("\n📊 Evaluating on validation data...")
val_loss, val_acc = model.evaluate(X_val_padded, y_val, verbose=0)
print(f"✅ Validation Accuracy: {val_acc:.4f}")
print(f"✅ Validation Loss: {val_loss:.4f}")


# === Evaluation on Test Data ===
print("\n📊 Evaluating on test data...")
test_loss, test_acc = model.evaluate(X_test_padded, y_test, verbose=0)
print(f"✅ Test Accuracy: {test_acc:.4f}")
print(f"✅ Test Loss: {test_loss:.4f}")

y_test_probs = model.predict(X_test_padded).flatten()
y_test_pred = (y_test_probs > 0.5).astype(int)

print("\n=== Test Data Evaluation Report ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=["Benign", "Malicious"]))


# === External Test Dataset (Optional) ===
if os.path.exists(TEST_DATA_PATH):
    print(f"\n📊 Loading external test dataset from {TEST_DATA_PATH}...")
    try:
        df_test_separate = pd.read_csv(TEST_DATA_PATH)
        df_test_separate["Payload"] = df_test_separate["Payload"].astype(str).apply(clean_payload)
        df_test_separate["Label"] = df_test_separate["Label"].astype(int)
        df_test_separate = df_test_separate[df_test_separate["Label"].isin([0, 1])]

        print(f"✅ External test dataset loaded: {len(df_test_separate)} samples")

        X_test_sep_seq = tokenizer.texts_to_sequences(df_test_separate["Payload"])
        X_test_sep_padded = pad_sequences(X_test_sep_seq, maxlen=MAX_LEN)
        y_test_sep = np.array(df_test_separate["Label"])

        test_sep_loss, test_sep_acc = model.evaluate(X_test_sep_padded, y_test_sep, verbose=0)
        print(f"✅ External Test Accuracy: {test_sep_acc:.4f}")
        print(f"✅ External Test Loss: {test_sep_loss:.4f}")

        y_test_sep_probs = model.predict(X_test_sep_padded).flatten()
        y_test_sep_pred = (y_test_sep_probs > 0.5).astype(int)

        # print("\n=== External Test Data Evaluation Report ===")
        # print("Confusion Matrix:")
        # print(confusion_matrix(y_test_sep, y_test_sep_pred))
        # print("\nClassification Report:")
        # print(classification_report(y_test_sep, y_test_sep_pred, target_names=["Benign", "Malicious"]))

    except Exception as e:
        print(f"❌ Error loading external test dataset: {e}")
else:
    print(f"\n⚠️ External test dataset not found at {TEST_DATA_PATH}")
    print("   Using internal test split for final evaluation")


# === Visualization: Label Distribution ===
train_counts = pd.Series(y_train).value_counts().sort_index()
val_counts = pd.Series(y_val).value_counts().sort_index()
test_counts = pd.Series(y_test).value_counts().sort_index()

print("\n=== Jumlah Payload per Label ===")
print(f"Train:      Benign = {train_counts.get(0,0)}, Malicious = {train_counts.get(1,0)}")
print(f"Validation: Benign = {val_counts.get(0,0)}, Malicious = {val_counts.get(1,0)}")
print(f"Test:       Benign = {test_counts.get(0,0)}, Malicious = {test_counts.get(1,0)}")

labels = ["Benign", "Malicious"]
x = np.arange(len(labels))
bar_width = 0.25

plt.figure(figsize=(8, 6))
plt.bar(x - bar_width, train_counts, width=bar_width, label="Train")
plt.bar(x, val_counts, width=bar_width, label="Validation")
plt.bar(x + bar_width, test_counts, width=bar_width, label="Test")

plt.xticks(x, labels)
plt.ylabel("Number of Payloads")
plt.title("Label Distribution (Train, Validation, Test)")
plt.legend()
plt.tight_layout()
plt.savefig("pict_label_distribution.png", dpi=300)
print("📊 Label distribution visualization saved as 'pict_label_distribution.png'")


# === Save Model ===
model.save(MODEL_PATH)
print(f"\n✅ Model saved to {MODEL_PATH}")


# === Visualization: Training Results ===
plt.figure(figsize=(15, 5))

# Accuracy
plt.subplot(1, 3, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy", marker="o")
plt.plot(history.history["val_accuracy"], label="Val Accuracy", marker="x")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 3, 2)
plt.plot(history.history["loss"], label="Train Loss", marker="o")
plt.plot(history.history["val_loss"], label="Val Loss", marker="x")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Confusion Matrix
plt.subplot(1, 3, 3)
cm = confusion_matrix(y_test, y_test_pred)
plt.imshow(cm, cmap="Blues", interpolation="nearest")
plt.title("Test Data Confusion Matrix")
plt.colorbar()
plt.xticks(ticks=[0, 1], labels=["Benign", "Malicious"])
plt.yticks(ticks=[0, 1], labels=["Benign", "Malicious"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]),
                 ha="center", va="center",
                 fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig("pict_training_results.png", dpi=300, bbox_inches="tight")
print("📊 Training visualization saved as 'pict_training_results.png'")
