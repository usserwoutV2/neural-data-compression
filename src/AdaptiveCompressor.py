import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
from tqdm import tqdm  
import random
import numpy as np
import time
import lzma

from encoders.Encoder import Encoder, AdaptiveEncoder
from SupporterModel import SupporterModel
from util.stats import show_plot
from util.util import  load_dataset,calculate_entropy, calculate_entropy_list
from util.match_string import match_string

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
    def __init__(self, hidden_size: int = 64, initial_learning_rate: float = 0.1, min_learning_rate: float = 0.001, decay_rate: float = 0.99, batch_size=128, encode_method="arithmetic", input_type="utf8"):
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
        self.encode_method = encode_method
        self.vocab_size = 128 if input_type == "utf8" else 256
        self.input_type = input_type
        self.use_rnn = False
        self.sequence_length = 32 


    def start_encoding(self):
        self.supporter_model = SupporterModel(self.hidden_size, self.hidden_size, self.vocab_size, quantize=False , use_rnn=self.use_rnn)
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
        if isinstance(input_string, bytes):
            input_string = input_string.decode('latin1')
        #self._create_vocabulary(input_string)
        self.start_encoding()
        input_indices = self._string_to_indices(input_string)
        num_samples = len(input_indices)
        compressed_indices = []

        adaptive_encoder = AdaptiveEncoder(self.vocab_size, method=self.encode_method)
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

            sorted_probs, sorted_indices = torch.sort(output, dim=1, descending=True)
            batch_inputs_tensor = torch.tensor(batch_inputs, dtype=torch.long)
            mask = sorted_indices == batch_inputs_tensor.unsqueeze(1)
            
            # Find the rank positions
            ranks = torch.argmax(mask.int(), dim=1).tolist()
            
            compressed_indices.extend(ranks)
            
            # Encode each rank
            for rank in ranks:
                adaptive_encoder.encode_symbol(rank)

                    
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
        return self.vocab_size.to_bytes(2, byteorder='big') + input_size_bytes + prefix + encoded_data


    def decompress(self, compressed_data: bytes) -> str:
        self.vocab_size = int.from_bytes(compressed_data[:2], byteorder='big')
        self.start_encoding()
        input_size = int.from_bytes(compressed_data[2:10], byteorder='big')
        prefix_indices = list(compressed_data[10:10 + self.sequence_length])
        decompressed_indices = prefix_indices.copy()
        adaptive_decoder = AdaptiveEncoder(self.vocab_size, method=self.encode_method)
        encoded_data = compressed_data[10 + self.sequence_length:]
        num_symbols = input_size - self.sequence_length
        decoded_ranks = adaptive_decoder.decode(encoded_data, num_symbols)

        batch_input_sequences = []
        batch_targets = []
        

        for i in tqdm(range(num_symbols), desc="Decompressing"):
            input_sequence = decompressed_indices[-self.sequence_length:]
            input_tensor = torch.tensor([input_sequence], dtype=torch.long)

            with torch.no_grad():
                output = self.supporter_model(input_tensor)
                probs = torch.softmax(output[:, -1, :], dim=1)
                target_index = torch.topk(probs, decoded_ranks[i] + 1, dim=1).indices[0, -1].item()
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
        
        if self.input_type == "bytes":
            return decompressed_string[:input_size].encode('latin1')
        return decompressed_string[:input_size]
        


compression_method = "arithmetic"
input_type = "utf8"
input_string = load_dataset("bible",1000_000)

# Model | execution time | output size (bytes, 100KB input size)
# LSMT 27 -> 43868
# Residual 13 -> 45457
# GRU 36 -> 42876
# Transformer 112.35 -> 45884
# Nothing 9 -> 46511
# RNN 1 hidden layer  13.61 -> 42774
# RNN 2 hidden layers 16.82 -> 42019
# RNN 3 hidden layers 19.74 -> 42710
# Self attention 208.51 -> 45467


def main():
    
    print(f"Original data size: {len(input_string)} bytes")
    #show_plot(input_string)
    print("Entropy",calculate_entropy(input_string))
    start_time = time.time()

    compressor = AdaptiveCompressor(hidden_size=64, initial_learning_rate=0.001, min_learning_rate=0.00005, decay_rate=0.9999, encode_method=compression_method, input_type=input_type, batch_size=128)
    compressed_data = compressor.compress(input_string)
    print(f"Compression took {time.time() - start_time:.2f} seconds")
    print(f"Compressed data size: {len(compressed_data)} bytes")

    # start_time = time.time()
    # decompressor = AdaptiveCompressor(hidden_size=64, initial_learning_rate=0.001, min_learning_rate=0.00005, decay_rate=0.9999, encode_method=compression_method, input_type=input_type, batch_size=128)
    # decompressed_string = decompressor.decompress(compressed_data)
    # print(f"Decompression took {time.time() - start_time:.2f} seconds")

    
    # if match_string(input_string, decompressed_string):
    #     print("Decompression successful!")

def compress_without_model():
    
    encoder = Encoder(method=compression_method)
    compressed = encoder._encode_str(input_string)
    print(f"Compressed data size (no support model): {len(compressed)} bytes")

if __name__ == "__main__":
    main()
    compress_without_model()