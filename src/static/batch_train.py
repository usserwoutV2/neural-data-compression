import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import itertools
import csv
from train import create_sequences, save_model

# Define the parameter values
sequence_length = [10, 30, 50]
epochs = [1, 3, 6]
batch_size = [32, 256, 1024]
lstm = [32, 128, 256]
embedding_dimension = [16, 64, 256]
patience = [1, 5, 10]
learning_rate = [0.001, 0.01, 0.1]


# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Define the dataset path
dataset_path = os.path.join(os.environ['VSC_HOME'], 'ML-project/datasets/data/bsb_small.txt')

# Generate all permutations
permutations = list(itertools.product(sequence_length, epochs, batch_size, lstm, embedding_dimension, patience, learning_rate))

# Convert permutations to a list of dictionaries
parameter_combinations = []
for perm in permutations:
    param_dict = {
        'sequence_length': perm[0],
        'epochs': perm[1],
        'batch_size': perm[2],
        'lstm': perm[3],
        'embedding_dimension': perm[4],
        'patience': perm[5],
        'learning_rate': perm[6]
    }
    parameter_combinations.append(param_dict)

# Read the dataset
with open(dataset_path, 'r') as file:
    input_string = file.read()

# Prepare the CSV file for logging results
csv_file = 'model_training_results_batch.csv'
csv_columns = ['sequence_length', 'epochs', 'batch_size', 'embedding_dimension', 'lstm', 'patience', 'learning_rate', 'accuracy']

with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()

    for params in parameter_combinations:
        # Train the model with the current parameter combination
        X, y, char_to_index, index_to_char = create_sequences(input_string, params['sequence_length'])
        
        vocab_size = len(char_to_index)
        
        model = Sequential([
            Embedding(vocab_size, params['embedding_dimension'], input_length=params['sequence_length']),
            LSTM(params['lstm'], return_sequences=True),
            Flatten(),
            Dense(params['lstm'], activation='relu'),
            Dense(vocab_size, activation='softmax'),
        ])
        
        optimizer = Adam(learning_rate=params['learning_rate'])
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Split data manually for training and validation
        split_at = int(0.9 * len(X))  # 90% training, 10% validation
        X_train, X_val = X[:split_at], X[split_at:]
        y_train, y_val = y[:split_at], y[split_at:]
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        
        train_dataset = train_dataset.shuffle(10000).batch(params['batch_size']).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(params['batch_size']).prefetch(tf.data.AUTOTUNE)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=params['patience'], restore_best_weights=True)
        
        history = model.fit(train_dataset, epochs=params['epochs'], validation_data=val_dataset, callbacks=[early_stopping], verbose=1)
        
        # Calculate accuracy
        accuracy = history.history['val_accuracy'][-1]

        # Save the model and lookup dictionaries
        model_name = f"model_seq{params['sequence_length']}_emb{params['embedding_dimension']}_lstm{params['lstm']}_pat{params['patience']}_lr{params['learning_rate']}.keras"
        lookup_name = f"lookup_seq{params['sequence_length']}_emb{params['embedding_dimension']}_lstm{params['lstm']}_pat{params['patience']}_lr{params['learning_rate']}.pkl"
        #save_model(model, char_to_index, index_to_char, model_path=model_name, lookup_path=lookup_name)

        # Calculate compression ratio (dummy value for now, replace with actual calculation)
        #compression_ratio = 1.0  # Replace with actual compression ratio calculation

        # Log the results to the CSV file
        result = {
            'sequence_length': params['sequence_length'],
            'epochs': params['epochs'],
            'batch_size': params['batch_size'],
            'embedding_dimension': params['embedding_dimension'],
            'lstm': params['lstm'],
            'learning_rate': params['learning_rate'],
			'patience': params['patience'],
            'accuracy': accuracy,
            #'compression_ratio': compression_ratio
        }
        writer.writerow(result)
        csvfile.flush()

print("Batch training completed. Results saved to", csv_file)