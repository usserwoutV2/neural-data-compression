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
from DictionarySupporterModel import DictionarySupporterModel
from exampleData import sample1, sample4
from stats import show_plot
import zlib
import lzma

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



def compress_array(array, k=2):
    # Ensure the array is a numpy array
    array = np.array(array, dtype=np.uint8)
    
    # Calculate the number of elements per byte
    elements_per_byte = 8 // k
    
    # Pad the array to make its length a multiple of elements_per_byte
    if len(array) % elements_per_byte != 0:
        padding_length = elements_per_byte - (len(array) % elements_per_byte)
        array = np.pad(array, (0, padding_length), 'constant', constant_values=0)
    
    # Reshape the array to group elements_per_byte elements together
    array = array.reshape(-1, elements_per_byte)
    
    # Convert each group of elements_per_byte elements into a single byte
    buffer = np.zeros((array.shape[0],), dtype=np.uint8)
    for i in range(elements_per_byte):
        buffer |= (array[:, i] << (8 - k * (i + 1)))
    
    return buffer.tobytes()

def decompress_array(compressed_data, k=2):
    # Convert the bytes back to a numpy array
    buffer = np.frombuffer(compressed_data, dtype=np.uint8)
    
    # Calculate the number of elements per byte
    elements_per_byte = 8 // k
    
    # Unpack the bits from the buffer
    unpacked = np.zeros((buffer.size, elements_per_byte), dtype=np.uint8)
    for i in range(elements_per_byte):
        unpacked[:, i] = (buffer >> (8 - k * (i + 1))) & ((1 << k) - 1)
    
    # Flatten the array to get the original k-bit numbers
    array = unpacked.flatten()
    
    return array


class DictionaryBasedAdaptiveCompressor(Encoder):
    def __init__(self, hidden_size: int = 64, initial_learning_rate: float = 0.1, min_learning_rate: float = 0.001, decay_rate: float = 0.99, batch_size=8):
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

    def _create_vocabulary(self):
        self.alphabet = [chr(i) for i in range(128)]
        self.vocab_size = len(self.alphabet)


        self.supporter_model = DictionarySupporterModel(self.hidden_size, self.hidden_size, self.vocab_size, quantize=False)
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

        adaptive_encoder = AdaptiveEncoder(128, method="lz78")
        adaptive_encoder.start_encoding()

        # Include the initial sequence in the compressed data
        prefix_indices = input_indices[:self.sequence_length]
        prefix = bytes(prefix_indices)
        model_indexes = []
        del_me = []
        
        # Start processing from index equal to sequence_length
        for batch_start in tqdm(range(self.sequence_length, num_samples, self.batch_size), desc="Compressing"):
            batch_end = min(batch_start + self.batch_size, num_samples)
            batch_inputs = input_indices[batch_start:batch_end]

            # Prepare input sequences from previous characters
            input_sequences = []
            for i in range(batch_start - self.sequence_length, batch_end - self.sequence_length):
                input_sequence = input_indices[i:i + self.sequence_length]
                input_sequences.append(input_sequence)

            # Convert to tensor
            input_tensor = torch.tensor(input_sequences, dtype=torch.long)
            # Forward pass without updating the model
          
            with torch.no_grad():
                models = self.supporter_model.forward_chunk(input_tensor)
                model_probabilities = [torch.softmax(model[:, -1, :], dim=1) for model in models]
            
            # Process each sample in the batch
            for j in range(batch_end - batch_start):
                target_index = batch_inputs[j]
                next_best_15 = adaptive_encoder.get_next_best()
                model_idx = -1
                final_rank = -1
                curr_place = 100
                for i, batch_probs in enumerate(model_probabilities):
                    probs = batch_probs[j]
                    target_prob = probs[target_index].item()
                    rank = (probs > target_prob).sum().item()
                    
                    if model_idx == -1 and i == len(model_probabilities) -1:
                        model_idx = i
                        final_rank = rank
                        curr_place = next_best_15.index(rank) if rank in next_best_15 else -1
                        continue
                    
                    index_in_best_array = next_best_15.index(rank) if rank in next_best_15 else -1
                    

                    if index_in_best_array != -1 and index_in_best_array < curr_place:
                        final_rank = rank
                        model_idx = i
                        curr_place = index_in_best_array

                del_me.append(final_rank)
                adaptive_encoder.encode_symbol(final_rank)
                model_indexes.append(model_idx)

            # Now update the model with the current batch
            self.optimizer.zero_grad()
            output = self.supporter_model.forward(input_tensor)
            output = output[:, -1, :]
            targets = torch.tensor(batch_inputs, dtype=torch.long)
            loss = self.criterion(output, targets)
            loss.backward()
            self.optimizer.step()

            self.current_step += 1
            self._update_learning_rate()

        encoded_data = adaptive_encoder.finish_encoding()
        input_size_bytes = len(input_string).to_bytes(8, byteorder='big')
        model_indexes_compressed = compress_array(np.array(model_indexes),k=1)
        
        model_indexes_compressed_size = len(model_indexes_compressed).to_bytes(4, byteorder='big')
        show_plot(del_me)
        show_plot(model_indexes)
        return input_size_bytes +model_indexes_compressed_size + model_indexes_compressed+ prefix + encoded_data 


    def decompress(self, compressed_data: bytes) -> str:
        self._create_vocabulary()
        input_size = int.from_bytes(compressed_data[:8], byteorder='big')
        model_indexes_size = int.from_bytes(compressed_data[8:12], byteorder='big')
        
        prefix_indices = list(compressed_data[12 + model_indexes_size :12 + self.sequence_length + model_indexes_size])
        model_indexes = decompress_array(compressed_data[12:12 + model_indexes_size])
        
        decompressed_indices = prefix_indices.copy()
        adaptive_decoder = AdaptiveEncoder(128)
        encoded_data = compressed_data[12 + self.sequence_length + model_indexes_size:]
        num_symbols = input_size - self.sequence_length
        decoded_ranks = adaptive_decoder.decode(encoded_data, num_symbols)

        batch_input_sequences = []
        batch_targets = []

        for i in tqdm(range(num_symbols), desc="Decompressing"):
            input_sequence = decompressed_indices[-self.sequence_length:]
            input_tensor = torch.tensor([input_sequence], dtype=torch.long)

            model_index = model_indexes[i]
            with torch.no_grad():
                output = self.supporter_model.forward_chunk(input_tensor)[model_index]
                output = output[:, -1, :]
                probabilities = torch.softmax(output, dim=1).squeeze(0)
            target_index = torch.topk(probabilities, decoded_ranks[i] + 1).indices[-1].item()

            decompressed_indices.append(target_index)
            batch_input_sequences.append(input_sequence)
            batch_targets.append(target_index)

            if len(batch_input_sequences) == self.batch_size or i == num_symbols - 1:
                batch_input_tensor = torch.tensor(batch_input_sequences, dtype=torch.long)
                batch_target_tensor = torch.tensor(batch_targets, dtype=torch.long)

                self.optimizer.zero_grad()
                output = self.supporter_model.forward(batch_input_tensor)
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

   

def main():
    
    input_string = sample4[:10_000]
    print(f"Original data size: {len(input_string)} bytes")
    show_plot(input_string)
    
    start_time = time.time()

    compressor = DictionaryBasedAdaptiveCompressor(hidden_size=64, initial_learning_rate=0.001, min_learning_rate=0.00005, decay_rate=0.9999)
    compressed_data = compressor.compress(input_string)
    print(f"Compression took {time.time() - start_time:.2f} seconds")
    print(f"Compressed data size: {len(compressed_data)} bytes")
    # 50_000 -> 34742 | 30164
    # 20_000 -> 13245 | 12902          <>            14042 | 12902
    
    # start_time = time.time()
    # decompressor = DictionaryBasedCompressor(hidden_size=64, initial_learning_rate=0.001, min_learning_rate=0.00005, decay_rate=0.9999)
    # decompressed_string = decompressor.decompress(compressed_data)
    # print(f"Decompression took {time.time() - start_time:.2f} seconds")

    
    # if input_string != decompressed_string:
    #     print(input_string[:100])   
    #     print("--------------------")
    #     print(decompressed_string[:100])
    #     print("Strings do not match!")
    # else:
    #     print("Decompression successful!")

def compress_without_model():
    input_string = sample4[:10_000]
    
    encoder = Encoder(method="lz78")
    compressed = encoder._encode_str(input_string)
    print(f"Compressed data size (no support model): {len(compressed)} bytes")

if __name__ == "__main__":
    main()
    compress_without_model()