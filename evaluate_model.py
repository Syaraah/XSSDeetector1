import pandas as pd
import numpy as np

import joblib

from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing.text_cleaning import clean_payload

#Load data
df = pd.read_csv('XSS_dataset.csv')
df['Payload'] = df['Payload'].astype(str).apply(clean_payload)
df['Label'] = df['Label'].astype(int)
df = df[df['Label'].isin([0, 1])]  

tokenizer = joblib.load('models/tokenizer.pkl')
model = load_model('models/xss_lstm_model.h5')
max_len = 200

