import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import sys
import os
from SupporterModel import SupporterModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Encoder import Encoder


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
        # Train the model on the input string
        self._create_vocabulary(input_string)
        self.supporter_model = SupporterModel(self.hidden_size, self.hidden_size, self.vocab_size)
        self.optimizer = optim.Adam(self.supporter_model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Prepare input and target tensors
        input_indices = self._string_to_indices(input_string)
        input_tensor = torch.tensor(input_indices[:-1], dtype=torch.long).unsqueeze(0)
        target_tensor = torch.tensor(input_indices[1:], dtype=torch.long)

        # Training loop
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            output = self.supporter_model(input_tensor)
            loss = self.criterion(output.squeeze(0), target_tensor)
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")

    def compress(self, input_string: str) -> Tuple[bytes, List[float]]:
        # Compress the input string
        self.train(input_string)
        
        # Prepare input tensor
        input_indices = self._string_to_indices(input_string)
        input_tensor = torch.tensor(input_indices[:-1], dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            logits = self.supporter_model(input_tensor).squeeze(0)
        
        # Calculate probabilities
        probabilities = torch.softmax(logits, dim=1)
        compressed_indices = []
        for i, prob in enumerate(probabilities):
            _, sorted_indices = torch.sort(prob, descending=True)
            index = (sorted_indices == input_indices[i+1]).nonzero(as_tuple=True)[0].item()
            compressed_indices.append(index)

        # Calculate frequency of each index
        freq = [0] * self.vocab_size
        for index in compressed_indices:
            freq[index] += 1
            
        print("compress indices:  ",compressed_indices)
        encoded_data = self._arithmetic_encode(compressed_indices, freq)
        return encoded_data, freq

    def decompress(self, compressed_data: bytes, freq: List[float], output_size: int) -> str:
        # Decompress the data
        decoded_indices = self._arithmetic_decode(compressed_data, freq, output_size - 1)
        print("decompress indices:",decoded_indices)

        decompressed_indices = [0]  # Start with an arbitrary initial value
        for i in range(output_size - 1):
            input_tensor = torch.tensor(decompressed_indices, dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                logits = self.supporter_model(input_tensor).squeeze(0)[-1]
            
            # Calculate probabilities
            probs = torch.softmax(logits, dim=0)
            _, sorted_indices = torch.sort(probs, descending=True)
            decompressed_indices.append(sorted_indices[decoded_indices[i]].item())
        return self._indices_to_string(decompressed_indices)

# Example usage
def main():
    input_string = "AAGAAGATAGGCACTTTGTTACCCAAAAAACCACCCCTGAGT"
    compressor = DynamicCompressor(hidden_size=64, epochs=10)
    
    compressed_data, freq = compressor.compress(input_string)
    print(f"Compressed data size: {len(compressed_data)} bytes")
    
    decompressed_string = compressor.decompress(compressed_data, freq, len(input_string))
    print(f"Original string: {input_string}")
    print(f"Decompressed string: {decompressed_string}")
    print(f"Compression successful: {input_string == decompressed_string}")

if __name__ == "__main__":
    main()