import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import csv
from train import create_sequences, save_model

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Define the dataset path
dataset_path = os.path.join(os.environ['VSC_HOME'], 'ML-project/datasets/files_to_be_compressed/chr20_train2.txt')

# Read the dataset
with open(dataset_path, 'r') as file:
    input_string = file.read()

# Prepare the CSV file for logging results
csv_file = 'model_training_results_hyperopt.csv'
csv_columns = ['sequence_length', 'epochs', 'batch_size', 'embedding_dimension', 'lstm', 'patience', 'learning_rate', 'accuracy']

# Define the search space for hyperopt
space = {
    'sequence_length': hp.choice('sequence_length', [50, 100, 150]),
    'epochs': hp.choice('epochs', [10, 20, 30]),
    'batch_size': hp.choice('batch_size', [64, 128, 256]),
    'embedding_dimension': hp.choice('embedding_dimension', [32, 64, 128]),
    'lstm': hp.choice('lstm', [64, 128, 256]),
    'patience': hp.choice('patience', [5, 10, 15]),
    'learning_rate': hp.choice('learning_rate', [0.0001, 0.001, 0.01]),
}

def objective(params):
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

    # Log the results to the CSV file
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        result = {
            'sequence_length': params['sequence_length'],
            'epochs': params['epochs'],
            'batch_size': params['batch_size'],
            'embedding_dimension': params['embedding_dimension'],
            'lstm': params['lstm'],
            'learning_rate': params['learning_rate'],
            'patience': params['patience'],
            'accuracy': accuracy,
        }
        writer.writerow(result)
    
    return {'loss': -accuracy, 'status': STATUS_OK}

# Run the hyperparameter optimization
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

print("Best hyperparameters found: ", best)
print("Hyperparameter optimization completed. Results saved to", csv_file)
