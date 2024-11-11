import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import sys
import os
from  SupporterModel import SupporterModel
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Encoder import Encoder
from exampleData import sample2,sample1,sample3,sample4
from stats import show_plot
from util import set_seed
import numpy as np


    
class DynamicCompressor(Encoder):
    def __init__(self, hidden_size: int = 64, learning_rate: float = 0.001, epochs: int = 10):
        # Initialize the DynamicCompressor with hyperparameters
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.char_to_index = {}
        self.index_to_char = {}
        self.vocab_size = 0

    def _create_vocabulary(self, input_string: str):
        # Create a vocabulary from the input string
        # This is used to convert between characters and indices
        unique_chars = sorted(set(input_string))
        self.vocab_size = len(unique_chars)
        self.char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
        self.index_to_char = {idx: char for idx, char in enumerate(unique_chars)}

    def _string_to_indices(self, input_string: str) -> List[int]:
        # Convert a string to a list of indices based on the vocabulary
        return [self.char_to_index[char] for char in input_string]

    def _indices_to_string(self, indices: List[int]) -> str:
        # Convert a list of indices back to a string based on the vocabulary
        return ''.join(self.index_to_char[idx] for idx in indices)

    def train(self, input_string: str):
        # Train the SupporterModel on the input string
        self._create_vocabulary(input_string)
        # Initialize the SupporterModel with the correct vocabulary size
        self.supporter_model = SupporterModel(self.hidden_size, self.hidden_size, vocab_size=self.vocab_size)
        self.optimizer = optim.Adam(self.supporter_model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Prepare input and target tensors
        input_indices = self._string_to_indices(input_string)
        input_tensor = torch.tensor(input_indices[:-1], dtype=torch.long).unsqueeze(0)
        target_tensor = torch.tensor(input_indices[1:], dtype=torch.long)

        # Training loop: update model parameters to minimize loss
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            output = self.supporter_model(input_tensor)
            loss = self.criterion(output.squeeze(0), target_tensor)
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")

    def compress(self, input_string: str) -> Tuple[bytes, List[float]]:
        # Compress the input string using the trained SupporterModel
        self.train(input_string)
        
        # Prepare input tensor
        input_indices = self._string_to_indices(input_string)
        
        input_tensor = torch.tensor(input_indices[:-1], dtype=torch.long).unsqueeze(0)
        # Use the SupporterModel to predict probabilities
        with torch.no_grad():
            logits = self.supporter_model(input_tensor).squeeze(0)
        
        # Calculate probabilities and compress based on predictions
        probabilities = torch.softmax(logits, dim=1)
        compressed_indices = []
        for i, prob in enumerate(probabilities):
            _, sorted_indices = torch.sort(prob, descending=True)
      
            index = (sorted_indices == input_indices[i+1]).nonzero(as_tuple=True)[0].item()
            compressed_indices.append(index)
        show_plot(compressed_indices)


        # Calculate frequency of each index for arithmetic coding
        freq = [0] * self.vocab_size
        for index in compressed_indices:
            freq[index] += 1
        
        #print("compress indices:  ", compressed_indices)
        # Use arithmetic coding to further compress the data
        first_char_index = input_indices[0]
        encoded_data = self._arithmetic_encode(compressed_indices, freq)
        return encoded_data, freq, first_char_index
    
    def decompress(self, compressed_data: bytes, freq: List[float], output_size: int, first_char_index: int) -> str:
        # Decompress the data using arithmetic decoding and the SupporterModel
        decoded_indices = self._arithmetic_decode(compressed_data, freq, output_size - 1)

        max_seq_len = 20  # Define a fixed maximum sequence length
        decompressed_indices = [first_char_index]  # Initialize the list to hold decompressed indices
        input_buffer = [first_char_index]
        
        # Initialize the input buffer with zeros or a start token
        input_buffer = [0] * (max_seq_len - 1) + input_buffer
        input_tensor = torch.tensor([input_buffer], dtype=torch.long)

        supporter_model = self.supporter_model
        supporter_model.eval()  # Set the model to evaluation mode

        for i in range(output_size - 1):
            # Use the SupporterModel to predict the next character
            with torch.no_grad():
                # Get the model's output for the last position
                logits = supporter_model(input_tensor)[0, -1]

            # Calculate probabilities and select the correct character
            probs = torch.softmax(logits, dim=0)
            _, sorted_indices = torch.sort(probs, descending=True)
            next_index = sorted_indices[decoded_indices[i]].item()
            decompressed_indices.append(next_index)

            # Update the input buffer with the new index
            input_buffer.append(next_index)
            # Keep only the last 'max_seq_len' elements
            input_buffer = input_buffer[-max_seq_len:]
            # Update the input tensor
            input_tensor = torch.tensor([input_buffer], dtype=torch.long)

        # Convert indices back to a string
        return self._indices_to_string(decompressed_indices)
    
    def save_compressed_data(self, model_save_path: str, compressed_data: bytes, freq: List[float], first_char_index: int):
        # Save the model's state dictionary and compressed data into one file
        data = {
            'model_state_dict': self.supporter_model.state_dict(),
            'compressed_data': compressed_data,
            'freq': freq,
            'char_to_index': self.char_to_index,
            'first_char_index': first_char_index,

        }
        with open(model_save_path, 'wb') as f:
            pickle.dump(data, f)
            


    def load_compressed_data(self, model_save_path: str):
        # Load the model's state dictionary and compressed data from one file
        with open(model_save_path, 'rb') as f:
            data = pickle.load(f)
        first_char_index = data['first_char_index']
        self.char_to_index = data['char_to_index']
        self.index_to_char = {idx: char for char, idx in self.char_to_index.items()}
        self.vocab_size = len(self.char_to_index)
        self.supporter_model = SupporterModel(self.hidden_size, self.hidden_size, self.vocab_size)
        self.supporter_model.load_state_dict(data['model_state_dict'])
        self.supporter_model.eval()
        
        compressed_data = data['compressed_data']
        freq = data['freq']
        
        return compressed_data, freq,first_char_index

# TODO:
# - Optimize stuff
# - Right now we save unnecessary data in the `save_compressed_data` function, like first_char_index. Find a better way.
def main():
    input_string = sample4 #[:10_000]
    set_seed(421)
    print(f"Original data size: {len(input_string)} bytes")
    show_plot(input_string)
    
    hidden_size = 58
    learning_rate = 0.0020963743367759346
    epochs = 20
    
    compressor = DynamicCompressor(hidden_size=hidden_size, epochs=epochs, learning_rate=learning_rate)
    
    start_time = time.time()
    compressed_data, freq, first_char_index = compressor.compress(input_string)
    print(f"Compression time: {time.time() - start_time:.2f} seconds")
    
    compressor.save_compressed_data("compressed_data.pkl", compressed_data, freq, first_char_index)
    print(f"Compressed data size: {len(compressed_data)} bytes")
    
    
    decompressor = DynamicCompressor(hidden_size=hidden_size, epochs=epochs, learning_rate=learning_rate)
    
    compressed_data, freq,first_char_index  = decompressor.load_compressed_data("compressed_data.pkl")

    print(f"Compressed data size: {len(compressed_data)} bytes")
    
    decompressed_string = decompressor.decompress(compressed_data, freq, len(input_string), first_char_index)
    if input_string != decompressed_string:
        print("Strings do not match!")
        print(input_string)
        print("-------------------------------------")
        print(decompressed_string)
    else:
        print("Decompression successful!")
    
    
    
def compress_without_model():
    input_string = sample4
    
    encoder = Encoder()
    compressed = encoder._arithmetic_encode_str(input_string)
    print(f"Compressed data size (no support model): {len(compressed)} bytes")


if __name__ == "__main__":
    main()
    #compress_without_model()
