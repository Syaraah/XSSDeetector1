import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from rensorflowkeras.callbacks import EarlyStopping
from text_cleaning import normalize_payload
import re

#Config
MAX_LEN = 200
EMBEDDING_DIM = 100

df = pd.read_csv("XSS_dataset.csv")
df['payload'] = df['payload'].astype(str)
df['label'] = df['label'].astype(int)
df = df[df['label'].isin([0, 1])]

#Distrubution check
print("Distribusi label sebelum balancing:")
print(df['label'].value_counts())

#Balancing dataset
from sklearn.utils import resample
df_majority = df[df['label'] == 0]
df_minority = df[df['label'] == 1]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df = pd.concat([df_majority, df_minority_upsampled]).sample(frac = 1).reset_index(drop=True)

with open('tokenizer/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

#Model
vocab_size = len(tokenizer.word_index) + 1

x_train = np.load('data/x_train.npy')
y_train = np.load('data/y_train.npy')
x_test = np.load('data/x_test.npy')
y_test = np.load('data/y_test.npy')

model = Sequential([
    Embedding(input_dim = vocab_size, output_dim = 128, input_length = max_len), 
    LSTM(64), 
    Dense(1, activation = 'sigmoid')
])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split = 0.2)

os.makedirs('models', exist_ok = True)
model.save('models/xss_detection_model.h5')

#eval
y_pred = (model.predict(x_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred, target_names = ["Benign", "Malicious"]))
