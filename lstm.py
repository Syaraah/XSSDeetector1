import os
import sys
import io
import datetime
import pydot
import graphviz

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

def build_lstm_model(vocab_size, max_len=200):
    model = Sequential()
    
    # Embedding layer with regularization
    model.add(Embedding(input_dim=vocab_size, output_dim=128))
    
    # First LSTM layer with more regularization
    model.add(LSTM(64, return_sequences=True, 
                   kernel_regularizer=l2(0.01),
                   recurrent_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))  
    
    # Second LSTM layer
    model.add(LSTM(32, 
                   kernel_regularizer=l2(0.01),
                   recurrent_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))  
    
    # Dense layer with regularization
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    VOCAB_SIZE = 5000
    MAX_LEN = 200

    model = build_lstm_model(VOCAB_SIZE, max_len=MAX_LEN)
    model.build(input_shape=(None, MAX_LEN))

    os.environ["GRAPHVIZ_DOT"] = r"C:\Users\Syarahil M\Downloads\Graphviz-12.2.0-win64\bin\dot.exe"
    os.environ["PATH"] += os.pathsep + r"C:\Users\Syarahil M\Downloads\Graphviz-12.2.0-win64\bin"

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    png_file = f"pict_lstm_architecture_{ts}.png"
    
    try:
        plot_model(
            model,
            to_file=png_file,
            show_shapes=True,
            show_layer_names=True,
            dpi=200
        )
        print(f"✅ Diagram tersimpan: {png_file}")
    except Exception as e:
        print("❌ Gagal menyimpan diagram")
        print("   Error:", e)
        