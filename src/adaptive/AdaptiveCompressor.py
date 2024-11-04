import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import sys
import os
from tqdm import tqdm  # Import tqdm for progress bar

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Encoder import Encoder, AdaptiveArithmeticEncoder
from dynamic.SupporterModel import SupporterModel
from exampleData import sample4
from stats import calculate_frequencies

class AdaptiveCompressor(Encoder):
    def __init__(self, hidden_size: int = 64, learning_rate: float = 0.001):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.char_to_index = {}
        self.index_to_char = {}
        self.vocab_size = 0
        self.supporter_model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

    def _create_vocabulary(self, input_string: str):
        unique_chars = sorted(set(input_string))
        self.vocab_size = len(unique_chars)
        self.char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
        self.index_to_char = {idx: char for idx, char in enumerate(unique_chars)}
        self.supporter_model = SupporterModel(self.hidden_size, self.hidden_size, self.vocab_size)
        self.optimizer = optim.Adam(self.supporter_model.parameters(), lr=self.learning_rate)

    def _string_to_indices(self, input_string: str) -> List[int]:
        return [self.char_to_index[char] for char in input_string]

    def _indices_to_string(self, indices: List[int]) -> str:
        return ''.join(self.index_to_char[idx] for idx in indices)

    def compress(self, input_string: str) -> Tuple[bytes, List[float], dict, dict]:
        self._create_vocabulary(input_string)
        input_indices = self._string_to_indices(input_string)
        compressed_indices = [input_indices[0]]  # Start with first character
        freq = [1] * self.vocab_size
        
        adaptive_encoder = AdaptiveArithmeticEncoder(self.index_to_char)
        adaptive_encoder.start_encoding()
        
        for i in tqdm(range(len(input_indices) - 1), desc="Compressing"):
            current_index = input_indices[i]
            target_index = input_indices[i + 1]
            input_tensor = torch.tensor([[current_index]], dtype=torch.long)
            
            # Forward pass
            output = self.supporter_model(input_tensor)
            probabilities = torch.softmax(output.squeeze(0), dim=1)
            
            # Get the rank of target_index in sorted probabilities
            probs_and_indices = list(enumerate(probabilities[0].tolist()))
            probs_and_indices.sort(key=lambda x: x[1], reverse=True)
            rank = next(i for i, (idx, _) in enumerate(probs_and_indices) if idx == target_index)
            compressed_indices.append(rank)
            
            # Train on actual next character
            loss = self.criterion(output.squeeze(0), torch.tensor([target_index]))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            freq[target_index] += 1
            adaptive_encoder.encode_symbol(self.index_to_char[target_index])

        encoded_data = adaptive_encoder.finish_encoding()
        return encoded_data, freq, self.char_to_index, self.index_to_char

    def decompress(self, compressed_data: bytes, freq: List[float], char_to_index: dict, index_to_char: dict, output_size: int) -> str:
        self.char_to_index = char_to_index
        self.index_to_char = index_to_char
        self.vocab_size = len(self.char_to_index)
        self.supporter_model = SupporterModel(self.hidden_size, self.hidden_size, self.vocab_size)
        self.optimizer = optim.Adam(self.supporter_model.parameters(), lr=self.learning_rate)
        
        adaptive_decoder = AdaptiveArithmeticEncoder(self.index_to_char)
        decoded_indices = adaptive_decoder.decode(compressed_data, output_size)
        decompressed_indices = [decoded_indices[0]]  # Start with first character
        
        for i in range(1, len(decoded_indices)):
            current_index = decompressed_indices[-1]
            input_tensor = torch.tensor([[current_index]], dtype=torch.long)
            
            # Forward pass
            output = self.supporter_model(input_tensor)
            probabilities = torch.softmax(output.squeeze(0), dim=1)
            
            # Get the character at rank decoded_indices[i]
            probs_and_indices = list(enumerate(probabilities[0].tolist()))
            probs_and_indices.sort(key=lambda x: x[1], reverse=True)
            next_index = probs_and_indices[decoded_indices[i]][0]
            decompressed_indices.append(next_index)
            
            # Train on predicted character
            loss = self.criterion(output.squeeze(0), torch.tensor([next_index]))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return self._indices_to_string(decompressed_indices)

# Example usage
def main():
    input_string = sample4[:1_000]
    print(f"Original data size: {len(input_string)} bytes")
    calculate_frequencies(input_string)

    compressor = AdaptiveCompressor(hidden_size=64, learning_rate=0.005)
    compressed_data, freq, char_to_index, index_to_char = compressor.compress(input_string)
    print(f"Compressed data size: {len(compressed_data)} bytes")

    decompressor = AdaptiveCompressor(hidden_size=64, learning_rate=0.005)
    decompressed_string = decompressor.decompress(compressed_data, freq, char_to_index, index_to_char, len(input_string))
    
    print(input_string )
    print("--------------------")
    print(decompressed_string)
    
    if input_string != decompressed_string:
        print("Strings do not match!")
    else:
        print("Decompression successful!")

    
def compress_without_model():
    input_string = sample4[:1_000]
    
    encoder = Encoder()
    compressed = encoder._arithmetic_encode_str(input_string)
    print(f"Compressed data size (no support model): {len(compressed)} bytes")
  


if __name__ == "__main__":
    main()
