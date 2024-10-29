import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import pickle
import os
import sys

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def create_sequences(text, sequence_length):
    char_to_index = {char: idx for idx, char in enumerate(sorted(set(text)))}
    index_to_char = {idx: char for char, idx in char_to_index.items()}
    
    input_sequences = []
    output_chars = []
    for i in range(0, len(text) - sequence_length):
        input_sequences.append(text[i:i+sequence_length])
        output_chars.append(text[i+sequence_length])
    
    X = np.array([[char_to_index[char] for char in seq] for seq in input_sequences])
    y = np.array([char_to_index[char] for char in output_chars])
    
    return X, y, char_to_index, index_to_char

def create_and_train_model(text, sequence_length=20, epochs=2, batch_size=64, lstm=128, embedding_dimension=16):
    X, y, char_to_index, index_to_char = create_sequences(text, sequence_length)
    
    vocab_size = len(char_to_index)
    
    model = Sequential([
        Embedding(vocab_size, embedding_dimension, input_length=sequence_length),
        LSTM(lstm, return_sequences=True),
        LSTM(lstm),
        Dense(vocab_size, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.01)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Split data manually for training and validation
    split_at = int(0.9 * len(X))  # 90% training, 10% validation
    X_train, X_val = X[:split_at], X[split_at:]
    y_train, y_val = y[:split_at], y[split_at:]
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    
    train_dataset = train_dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[early_stopping], verbose=1)
    
    return model, char_to_index, index_to_char

def predict_next_chars(model, char_to_index, index_to_char, input_string, sequence_length=20):
    input_indices = np.array([[char_to_index.get(char, 0) for char in input_string[-sequence_length:]]])
    predictions = model.predict(input_indices, verbose=0)[0]
    top_indices = predictions.argsort()[-4:][::-1]
    predicted_chars = [index_to_char[idx] for idx in top_indices]
    return predicted_chars

def save_model(model, char_to_index, index_to_char, model_path='char_model.keras', lookup_path='char_lookup.pkl'):
    model.save(model_path)
    with open(lookup_path, 'wb') as handle:
        pickle.dump((char_to_index, index_to_char), handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
def load_model(model_path='char_model.keras', lookup_path='char_lookup.pkl'):
    model = keras_load_model(model_path)
    with open(lookup_path, 'rb') as handle:
        char_to_index, index_to_char = pickle.load(handle)
    return model, char_to_index, index_to_char