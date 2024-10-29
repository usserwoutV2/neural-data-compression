import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import sys
import os
from  SupporterModel import SupporterModel
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Encoder import Encoder
from exampleData import sample2,sample1,sample3,sample4
from stats import calculate_frequencies

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
        self.supporter_model = SupporterModel(self.hidden_size, self.hidden_size, self.vocab_size)
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
        calculate_frequencies(compressed_indices)

        # Calculate frequency of each index for arithmetic coding
        freq = [0] * self.vocab_size
        for index in compressed_indices:
            freq[index] += 1
            
        #print("compress indices:  ", compressed_indices)
        # Use arithmetic coding to further compress the data
        encoded_data = self._arithmetic_encode(compressed_indices, freq)
        return encoded_data, freq
    
    def decompress(self, compressed_data: bytes, freq: List[float], output_size: int) -> str:
        # Decompress the data using arithmetic decoding and the SupporterModel
        decoded_indices = self._arithmetic_decode(compressed_data, freq, output_size - 1)
        #print("decompress indices:", decoded_indices)

        decompressed_indices = [0]  # Start with an arbitrary initial value
        input_tensor = torch.tensor(decompressed_indices, dtype=torch.long).unsqueeze(0)
        supporter_model = self.supporter_model

        for i in range(output_size - 1):
            # Use the SupporterModel to predict the next character
            with torch.no_grad():
                logits = supporter_model(input_tensor).squeeze(0)[-1]
                            
            # Calculate probabilities and select the correct character
            probs = torch.softmax(logits, dim=0)
            _, sorted_indices = torch.sort(probs, descending=True)
            next_index = sorted_indices[decoded_indices[i]].item()
            decompressed_indices.append(next_index)
            
            # Update the input tensor in place
            input_tensor = torch.cat((input_tensor, torch.tensor([[next_index]], dtype=torch.long)), dim=1)

        # Convert indices back to a string
        return self._indices_to_string(decompressed_indices)
    
    def save_compressed_data(self, model_save_path: str, compressed_data: bytes, freq: List[float]):
        # Save the model's state dictionary and compressed data into one file
        data = {
            'model_state_dict': self.supporter_model.state_dict(),
            'compressed_data': compressed_data,
            'freq': freq,
            'char_to_index': self.char_to_index
        }
        with open(model_save_path, 'wb') as f:
            pickle.dump(data, f)
            


    def load_compressed_data(self, model_save_path: str):
        # Load the model's state dictionary and compressed data from one file
        with open(model_save_path, 'rb') as f:
            data = pickle.load(f)
        
        self.char_to_index = data['char_to_index']
        self.index_to_char = {idx: char for char, idx in self.char_to_index.items()}
        self.vocab_size = len(self.char_to_index)
        self.supporter_model = SupporterModel(self.hidden_size, self.hidden_size, self.vocab_size)
        self.supporter_model.load_state_dict(data['model_state_dict'])
        self.supporter_model.eval()
        
        compressed_data = data['compressed_data']
        freq = data['freq']
        
        return compressed_data, freq

# Example usage
def main():
    input_string = sample4
    print(f"Original data size: {len(input_string)} bytes")
    calculate_frequencies(input_string)
    
    compressor = DynamicCompressor(hidden_size=64, epochs=50, learning_rate=0.005)
    
    compressed_data, freq = compressor.compress(input_string)
    compressor.save_compressed_data("compressed_data.pkl", compressed_data, freq)
    print(f"Compressed data size: {len(compressed_data)} bytes")
    
    
    # decompressor = DynamicCompressor(hidden_size=64, epochs=20, learning_rate=0.005)
    
    # compressed_data, freq  = decompressor.load_compressed_data("compressed_data.pkl")

    # print(f"Compressed data size: {len(compressed_data)} bytes")
    
    # decompressed_string = decompressor.decompress(compressed_data, freq, len(input_string))
    # # print(f"Original string: {input_string}")
    # #print(f"Decompressed string: {decompressed_string}")
    # if input_string != decompressed_string:
    #     print("Strings do not match!")
    #     print(input_string)
    #     print("-------------------------------------")
    #     print(decompressed_string)
    
    
def compress_without_model():
    input_string = sample4
    
    encoder = Encoder()
    compressed = encoder._arithmetic_encode_str(input_string)
    print(f"Compressed data size (no support model): {len(compressed)} bytes")
    

if __name__ == "__main__":
    main()
    compress_without_model()