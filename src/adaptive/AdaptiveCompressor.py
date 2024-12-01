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
from Encoder import Encoder, AdaptiveEncoder
from dynamic.SupporterModel import SupporterModel
from exampleData import sample1, sample4
from stats import show_plot
from util import  load_dataset

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
    def __init__(self, hidden_size: int = 64, initial_learning_rate: float = 0.1, min_learning_rate: float = 0.001, decay_rate: float = 0.99, batch_size=8, encode_method="arithmetic"):
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
        self.batch_size = batch_size
        self.sequence_length = 32 
        self.encode_method = encode_method

    def _create_vocabulary(self):
        self.alphabet = [chr(i) for i in range(128)]
        self.vocab_size = len(self.alphabet)


        self.supporter_model = SupporterModel(self.hidden_size, self.hidden_size, self.vocab_size, quantize=False)
        self.optimizer = optim.Adam(self.supporter_model.parameters(), lr=self.initial_learning_rate)

    def _update_learning_rate(self):
        new_lr = max(self.min_learning_rate, self.initial_learning_rate * (self.decay_rate ** (self.current_step )))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def _string_to_indices(self, input_string: str) -> List[int]:
        return [ord(char) for char in input_string]

    def _indices_to_string(self, indices: List[int]) -> str:
        return ''.join(chr(idx) for idx in indices)

    

    def compress(self, input_string: str) -> bytes:
        self._create_vocabulary()
        input_indices = self._string_to_indices(input_string)
        num_samples = len(input_indices)
        compressed_indices = []

        adaptive_encoder = AdaptiveEncoder(128, method=self.encode_method)
        adaptive_encoder.start_encoding()

        # Include the initial sequence in the compressed data
        prefix_indices = input_indices[:self.sequence_length]
        prefix = bytes(prefix_indices)

        # Start processing from index equal to sequence_length
        for batch_start in tqdm(range(self.sequence_length, num_samples, self.batch_size), desc="Compressing"):
            batch_end = min(batch_start + self.batch_size, num_samples)
            batch_inputs = input_indices[batch_start:batch_end]
            
            input_sequences = [input_indices[i:i + self.sequence_length] for i in range(batch_start - self.sequence_length, batch_end - self.sequence_length)]
            input_tensor = torch.tensor(input_sequences, dtype=torch.long)
            output = self.supporter_model(input_tensor)
            
            output = output[:, -1, :]  # Shape: [batch_size, vocab_size]

            # Sort probabilities and get sorted indices for the batch
            sorted_probs, sorted_indices = torch.sort(output, dim=1, descending=True)
            
            # Convert batch_inputs to tensor and create a mask where sorted_indices match batch_inputs
            batch_inputs_tensor = torch.tensor(batch_inputs, dtype=torch.long)
            mask = sorted_indices == batch_inputs_tensor.unsqueeze(1)
            
            # Find the rank positions
            ranks = torch.argmax(mask.int(), dim=1).tolist()
            
            compressed_indices.extend(ranks)

            # Encode each rank
            for rank in ranks:
                adaptive_encoder.encode_symbol(rank)

            # Now update the model with the current batch
            self.optimizer.zero_grad()
            # output = self.supporter_model(input_tensor)
            # output = output[:, -1, :]
            targets = torch.tensor(batch_inputs, dtype=torch.long)
            loss = self.criterion(output, targets)
            loss.backward()
            self.optimizer.step()

            self.current_step += 1
            self._update_learning_rate()

        encoded_data = adaptive_encoder.finish_encoding()
        input_size_bytes = len(input_string).to_bytes(8, byteorder='big')

        # Return the size, prefix, and encoded data
        return input_size_bytes + prefix + encoded_data


    def decompress(self, compressed_data: bytes) -> str:
        self._create_vocabulary()
        input_size = int.from_bytes(compressed_data[:8], byteorder='big')
        prefix_indices = list(compressed_data[8:8 + self.sequence_length])
        decompressed_indices = prefix_indices.copy()
        adaptive_decoder = AdaptiveEncoder(128, method=self.encode_method)
        encoded_data = compressed_data[8 + self.sequence_length:]
        num_symbols = input_size - self.sequence_length
        decoded_ranks = adaptive_decoder.decode(encoded_data, num_symbols)

        batch_input_sequences = []
        batch_targets = []

        for i in tqdm(range(num_symbols), desc="Decompressing"):
            input_sequence = decompressed_indices[-self.sequence_length:]
            input_tensor = torch.tensor([input_sequence], dtype=torch.long)

            with torch.no_grad():
                output = self.supporter_model(input_tensor)
                
                target_index = torch.topk(output[:, -1, :], decoded_ranks[i] + 1, dim=1).indices[0, -1].item()
                
                decompressed_indices.append(target_index)
                batch_input_sequences.append(input_sequence)
                batch_targets.append(target_index)

            if len(batch_input_sequences) == self.batch_size or i == num_symbols - 1:
                batch_input_tensor = torch.tensor(batch_input_sequences, dtype=torch.long)
                batch_target_tensor = torch.tensor(batch_targets, dtype=torch.long)

                self.optimizer.zero_grad()
                output = self.supporter_model(batch_input_tensor)
                output = output[:, -1, :]
                loss = self.criterion(output, batch_target_tensor)
                loss.backward()
                self.optimizer.step()

                batch_input_sequences = []
                batch_targets = []

                self.current_step += 1
                self._update_learning_rate()

        decompressed_string = self._indices_to_string(decompressed_indices)
        return decompressed_string[:input_size]


compression_method = "arithmetic"

input_string = load_dataset("bible", 20_000)


def main():
    
    print(f"Original data size: {len(input_string)} bytes")
    show_plot(input_string)
    
    start_time = time.time()

    compressor = AdaptiveCompressor(hidden_size=64, initial_learning_rate=0.001, min_learning_rate=0.00005, decay_rate=0.9999, encode_method=compression_method)
    compressed_data = compressor.compress(input_string)
    print(f"Compression took {time.time() - start_time:.2f} seconds")
    print(f"Compressed data size: {len(compressed_data)} bytes")

    start_time = time.time()
    decompressor = AdaptiveCompressor(hidden_size=64, initial_learning_rate=0.001, min_learning_rate=0.00005, decay_rate=0.9999, encode_method=compression_method)
    decompressed_string = decompressor.decompress(compressed_data)
    print(f"Decompression took {time.time() - start_time:.2f} seconds")

    
    if input_string != decompressed_string:
        print(input_string[:100])   
        print("--------------------")
        print(decompressed_string[:100])
        print("Strings do not match!")
    else:
        print("Decompression successful!")

def compress_without_model():
    
    encoder = Encoder(method=compression_method)
    compressed = encoder._encode_str(input_string)
    print(f"Compressed data size (no support model): {len(compressed)} bytes")

if __name__ == "__main__":
    main()
    compress_without_model()