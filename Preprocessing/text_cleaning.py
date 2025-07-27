import pandas as pd
import urllib.parse 
import html
import pickle
import numpy as np
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

#Load the dataset
df = pd.read_csv("XSS_dataset.csv")

#Normalize payload
def normalize_payload(text):
    text = html.unescape(text) # Decode HTML entities
    text = urllib.parse.unquote(text) # Decode URL encoding
    return text

df['Sentence'] = df['Sentence'].apply(normalize_payload)

#Tokenization and padding
tokenizer = Tokenizer(char_level = False)
tokenizer.fit_on_texts(df['Sentence'])

sequences = tokenizer.texts_to_sequences(df['Sentence'])
max_len = 200 
x = pad_sequences(sequences, maxlen = max_len, padding = 'post')
y = df['Label'].values

#Save the tokenizer
os.makedirs('tokenizer', exist_ok=True)
with open('tokenizer/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
    
#Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Save data
os.makedirs('data', exist_ok=True)
np.save('data/x_train.npy', x_train)
np.save('data/x_test.npy', x_test)  
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)

print("[✓] Preprocessing selesai. Tokenizer dan data berhasil disimpan.")

