import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import sys
import os
from tqdm import tqdm  
import random
import numpy as np
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Encoder import Encoder, AdaptiveArithmeticEncoder
from dynamic.SupporterModel import SupporterModel
from exampleData import sample1
from stats import show_plot

def set_seed(seed: int):
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
    # Set the seed for NumPy
    np.random.seed(seed)
    
    # Set the seed for Python's built-in random module
    random.seed(seed)
    
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
class AdaptiveCompressor(Encoder):
    def __init__(self, hidden_size: int = 64, initial_learning_rate: float = 0.1, min_learning_rate: float = 0.001, decay_rate: float = 0.99):
        set_seed(42)
        self.hidden_size = hidden_size
        self.initial_learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.decay_rate = decay_rate
        self.current_step = 0
        self.char_to_index = {}
        self.index_to_char = {}
        self.vocab_size = 0
        self.supporter_model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.tensor_size = 1

    def _create_vocabulary(self):
        self.alphabet = [chr(i) for i in range(128)]
        self.vocab_size = len(self.alphabet)


        self.supporter_model = SupporterModel(self.hidden_size, self.hidden_size, self.vocab_size, quantize=False)
        self.optimizer = optim.Adam(self.supporter_model.parameters(), lr=self.initial_learning_rate)

    def _update_learning_rate(self):
        new_lr = max(self.min_learning_rate, self.initial_learning_rate * (self.decay_rate ** self.current_step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def _string_to_indices(self, input_string: str) -> List[int]:
        return [ord(char) for char in input_string]

    def _indices_to_string(self, indices: List[int]) -> str:
        return ''.join(chr(idx) for idx in indices)



    def compress(self, input_string: str) -> Tuple[bytes, List[float], dict, dict]:
        self._create_vocabulary()
        input_indices = self._string_to_indices(input_string)
        compressed_indices = [input_indices[0]]
        freq = [1] * self.vocab_size
        
        adaptive_encoder = AdaptiveArithmeticEncoder(128)
        adaptive_encoder.start_encoding()
        
        input_buffer = input_indices[:self.tensor_size]
        prefix = bytes(input_buffer)
        del_me = []
        
        for i in tqdm(range(self.tensor_size, len(input_indices)), desc="Compressing"):
            self.current_step = i
            self._update_learning_rate()
            
            current_indices = input_buffer[-self.tensor_size:]
            target_index = input_indices[i]
            input_tensor = torch.tensor([current_indices], dtype=torch.long)
            
            # Forward pass
            output = self.supporter_model(input_tensor)
            output_squeezed = output.squeeze(0)
            probabilities = torch.softmax(output_squeezed, dim=1)[0]
            
            target_prob = probabilities[target_index].item()
            
            # Count how many probabilities are higher than the target_prob
            rank = (probabilities > target_prob).sum().item()
            compressed_indices.append(rank)
            
            # Train on actual next character
            loss = self.criterion(output_squeezed, torch.tensor([target_index]))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            freq[target_index] += 1
            adaptive_encoder.encode_symbol(rank)
            
            # Update the input buffer
            input_buffer.append(target_index)
            del_me.append(rank)
            
        
        show_plot(del_me)
            
        encoded_data = adaptive_encoder.finish_encoding()
        # Append the size of the input to the output
        input_size_bytes = len(input_string).to_bytes(8, byteorder='big')

        return input_size_bytes + prefix + encoded_data

    def decompress(self, compressed_data: bytes) -> str:
        self._create_vocabulary()
        output_size = int.from_bytes(compressed_data[:8], byteorder='big')

        adaptive_decoder = AdaptiveArithmeticEncoder(128)
        decoded_indices = adaptive_decoder.decode(compressed_data[self.tensor_size + 8:], output_size - self.tensor_size)
        decompressed_indices = list(compressed_data[8:8+self.tensor_size])
        
        for i in tqdm(range(self.tensor_size, len(decoded_indices)+1), desc="Decompressing"):
            self.current_step = i
            self._update_learning_rate()
            
            current_indices = decompressed_indices[-self.tensor_size:]
            input_tensor = torch.tensor([current_indices], dtype=torch.long)
            
            # Forward pass
            output = self.supporter_model(input_tensor)
            probabilities = torch.softmax(output.squeeze(0), dim=1)[0]
            target_index = decoded_indices[i - self.tensor_size]
            # Get the index of the nth largest probability directly
            next_index = torch.topk(probabilities, target_index + 1).indices[-1].item()
 

            decompressed_indices.append(next_index)
            
            # Train on predicted character
            loss = self.criterion(output.squeeze(0), torch.tensor([next_index]))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return self._indices_to_string(decompressed_indices)

def main():
    
    input_string = sample1[:5_000]
    print(f"Original data size: {len(input_string)} bytes")
    show_plot(input_string)
    
    start_time = time.time()

    compressor = AdaptiveCompressor(hidden_size=64, initial_learning_rate=0.001, min_learning_rate=0.00005, decay_rate=0.9999)
    compressed_data = compressor.compress(input_string)
    print(f"Compression took {time.time() - start_time:.2f} seconds")
    print(f"Compressed data size: {len(compressed_data)} bytes")

    start_time = time.time()
    decompressor = AdaptiveCompressor(hidden_size=64, initial_learning_rate=0.001, min_learning_rate=0.00005, decay_rate=0.9999)
    decompressed_string = decompressor.decompress(compressed_data)
    print(f"Decompression took {time.time() - start_time:.2f} seconds")

    
    if input_string != decompressed_string:
        print(input_string)
        print("--------------------")
        print(decompressed_string)
        print("Strings do not match!")
    else:
        print("Decompression successful!")

def compress_without_model():
    input_string = sample1[:20_000]
    
    encoder = Encoder()
    compressed = encoder._arithmetic_encode_str(input_string)
    print(f"Compressed data size (no support model): {len(compressed)} bytes")

if __name__ == "__main__":
    main()
    compress_without_model()