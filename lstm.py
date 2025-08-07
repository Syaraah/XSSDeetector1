from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def build_lstm_model(vocab_size, max_len=200):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=128))  
    model.add(LSTM(64, return_sequences=True))  
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model
