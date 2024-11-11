
from DynamicCompressor import DynamicCompressor
import optuna

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from exampleData import sample2,sample1,sample3,sample4

from util import set_seed

def objective(trial):
    # Define the hyperparameters to optimize
    hidden_size = trial.suggest_int('hidden_size', 32, 128)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    epochs = trial.suggest_int('epochs', 10, 50)
    
    # Set seed for reproducibility
    set_seed(421)
    
    # Initialize the compressor with the trial's hyperparameters
    compressor = DynamicCompressor(hidden_size=hidden_size, learning_rate=learning_rate, epochs=epochs)
    
    # Sample input data
    input_string = sample4[:10_000]
    
    # Compress and decompress the input string
    compressed_data, freq, first_char_index = compressor.compress(input_string)
    decompressed_string = compressor.decompress(compressed_data, freq, len(input_string), first_char_index)
    
    # Calculate the loss as the number of mismatched characters
    loss = sum(1 for a, b in zip(input_string, decompressed_string) if a != b)
    
    return loss
  
  
  
if __name__ == "__main__":
    #main()
    #compress_without_model()
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    # Print the best hyperparameters
    print("Best hyperparameters: ", study.best_params)
    print("Best loss: ", study.best_value)
    