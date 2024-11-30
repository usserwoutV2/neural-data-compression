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
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Encoder import Encoder, AdaptiveEncoder
from dynamic.SupporterModel import SupporterModel
from exampleData import sample1, sample4
from stats import show_plot
from util import load_dataset

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
    def __init__(
        self,
        hidden_size: int = 64,
        initial_learning_rate: float = 0.1,
        min_learning_rate: float = 0.001,
        decay_rate: float = 0.99,
        batch_size=8,
        encode_method="arithmetic",
    ):
        set_seed(42)
        self.hidden_size = hidden_size
        self.initial_learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.decay_rate = decay_rate
        self.current_step = 0
        self.char_to_index = {}
        self.index_to_char = {}
        self.vocab_size = 1000
        self.supporter_model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.sequence_length = 32
        self.encode_method = encode_method
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()
        self.trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=["[PAD]", "[UNK]"], show_progress=False)  # Reduced vocab size

    def _create_vocabulary(self):
        self.supporter_model = SupporterModel(
            self.hidden_size, self.hidden_size, self.vocab_size, quantize=False
        )
        self.optimizer = optim.Adam(
            self.supporter_model.parameters(), lr=self.initial_learning_rate
        )

    def _update_learning_rate(self):
        new_lr = max(
            self.min_learning_rate,
            self.initial_learning_rate * (self.decay_rate ** (self.current_step)),
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    def _string_to_indices(self, input_string: List[str]) -> List[int]:
        print(self.tokenizer.encode(input_string).tokens)
        return self.tokenizer.encode(input_string).ids

    def _indices_to_string(self, indices: List[int]) -> str:
        return self.tokenizer.decode(indices)

    def _train_tokenizer(self, texts: List[str]):
        self.tokenizer.train_from_iterator(texts, self.trainer)

    def _string_to_ord(self, input_string: str) -> List[int]:
        return [ord(char) for char in input_string]
    
    def _ord_to_string(self, input_string: List[int]) -> List[str]:
        return [chr(char) for char in input_string]

    def compress(self, input_string: str) -> bytes:
        self._create_vocabulary()
        num_samples = len(input_string)
        compressed_indices = []

        adaptive_encoder = AdaptiveEncoder(self.vocab_size, method=self.encode_method)
        adaptive_encoder.start_encoding()

        # Include the initial sequence in the compressed data
        prefix_indices = input_string[: self.sequence_length]
        alphabet = "".join([chr(i) for i in range(128)])
        self._train_tokenizer([alphabet,prefix_indices])

        prefix_ind = self._string_to_ord(prefix_indices)
        prev_chunks = np.array(prefix_ind, dtype=int)
        prefix = bytes(prefix_ind)
        
        # Split input into chunks
        for batch_start in tqdm(
            range(self.sequence_length, num_samples, self.batch_size),
            desc="Compressing",
        ):
            batch_end = min(batch_start + self.batch_size, num_samples)
            batch_inputs = input_string[batch_start:batch_end]

            
            input_indices = self._string_to_indices(batch_inputs)
            print(f" {batch_inputs[:10]} -> {self._indices_to_string(input_indices)[:10]}")

            # Prepare input sequences from previous characters
            input_sequences = []
            for i in range(
                self.sequence_length, len(input_indices) + self.sequence_length
            ):
                if i  <= 2 * self.sequence_length:
                    input_sequence = np.concatenate((prev_chunks[i - self.sequence_length  :], input_indices[:i - self.sequence_length ]))
                else:
                    input_sequence = input_indices[i - 2*self.sequence_length  : i - self.sequence_length]
                input_sequences.append(input_sequence)
                
            
            if batch_start == self.sequence_length:
                print("previous chunk",  prev_chunks)
                print(f"Batch input: '{batch_inputs}'")
                print("INPUT: ", list(input_sequences[0]))
            
            # Convert to tensor
            input_tensor = torch.tensor(input_sequences, dtype=torch.long)
            # Forward pass without updating the model
            
            output = self.supporter_model(input_tensor)
            output = output[:, -1, :]  # Get output for the last time step
            probabilities = torch.softmax(output, dim=1).squeeze(0)

            # Process each sample in the batch
            for j, probs in enumerate(probabilities):
                target_index = input_indices[j]
                
                
                target_prob = probs[target_index].item()

                # Count how many probabilities are higher than the target_prob
                rank = (probs > target_prob).sum().item()
                
                if j == 0 and batch_start == self.sequence_length:
                    print(f"Target index: {target_index}, rank: {rank}")
                    print(probs[:20])
                compressed_indices.append(rank)

                adaptive_encoder.encode_symbol(rank)

            # Now update the model with the current batch
            self.optimizer.zero_grad()
            targets = torch.tensor(input_indices, dtype=torch.long)
            loss = self.criterion(output, targets)
            loss.backward()
            self.optimizer.step()
            
            shift_len = len(input_indices)
            if shift_len < self.sequence_length:
                prev_chunks = np.roll(prev_chunks, -shift_len)
                prev_chunks[-shift_len:] = input_indices[:shift_len]
            else:
                prev_chunks = input_indices[-self.sequence_length:]
            
            self.current_step += 1
            self._update_learning_rate()
            
            
            self._train_tokenizer([alphabet,input_string[:batch_end]])

        #show_plot(compressed_indices)
        encoded_data = adaptive_encoder.finish_encoding()
        input_size_bytes = len(compressed_indices).to_bytes(8, byteorder="big")

        # Return the size, prefix, and encoded data
        return input_size_bytes + prefix + encoded_data

    def decompress(self, compressed_data: bytes) -> str:
        self._create_vocabulary()
        input_size = int.from_bytes(compressed_data[:8], byteorder="big")
        prefix_indices = list(compressed_data[8 : 8 + self.sequence_length])
        
        decompressed_indices = prefix_indices.copy()
        decompressed_string = list(self._ord_to_string(prefix_indices))
        adaptive_decoder = AdaptiveEncoder(self.vocab_size, method=self.encode_method)
        encoded_data = compressed_data[8 + self.sequence_length :]
        num_symbols = input_size #- self.sequence_length
        decoded_ranks = adaptive_decoder.decode(encoded_data, num_symbols)
        
    
        batch_input_sequences = []
        batch_targets = []

        # Initialize the tokenizer with the prefix and alphabet
        alphabet = "".join([chr(i) for i in range(128)])
        self._train_tokenizer([alphabet, "".join(decompressed_string)])

        
        chunk = np.array(decompressed_indices, dtype=int)

        for i in tqdm(range(num_symbols), desc="Decompressing"):
            input_sequence = decompressed_indices[-self.sequence_length:]
            
           
            input_tensor = torch.tensor([input_sequence], dtype=torch.long)
            
                    
                        
            print("INPUT: ",input_sequence)
            
            output = self.supporter_model(input_tensor)

            probabilities = torch.softmax(output[:, -1, :], dim=1).squeeze(0)

            rank = torch.topk(probabilities, decoded_ranks[i] + 1, dim=0).indices[-1].item()
            

            character = self._indices_to_string([rank])
            
            print(f"==== Decoded character: {character}, rank: {rank} decoded_ranks[i]: {decoded_ranks[i]} =====")
            print(probabilities[:20])

            exit(1)
            decompressed_indices.append(rank)
            decompressed_string.append(character)
            
            batch_input_sequences.append(input_sequence)
            batch_targets.append(rank)


            np.roll(chunk, -1)
            chunk[-1] = decoded_ranks[i]
            
            if len(batch_input_sequences) == self.batch_size or i == num_symbols - 1:
                batch_input_tensor = torch.tensor(
                    batch_input_sequences, dtype=torch.long
                )
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

            # Update prev_chunks by shifting len(input_indices) to the left and adding new elements
            shift_len = 1
            prev_chunks = np.roll(prev_chunks, -shift_len)
            prev_chunks[-shift_len:] = [target_index]

            # Update the tokenizer with the current batch without resetting the vocabulary
            self._train_tokenizer([alphabet, "".join(decompressed_string)])

        return "".join(decompressed_string)


compression_method = "arithmetic"



input_string = load_dataset("bible", 10_000)

def main():
    print(f"Original data size: {len(input_string)} bytes")
    show_plot(input_string)

    start_time = time.time()

    compressor = AdaptiveCompressor(
        hidden_size=64,
        initial_learning_rate=0.001,
        min_learning_rate=0.00005,
        decay_rate=0.9999,
        encode_method=compression_method,
        batch_size=128
    )
    compressed_data = compressor.compress(input_string)
    print(f"Compression took {time.time() - start_time:.2f} seconds")
    print(f"Compressed data size: {len(compressed_data)} bytes")
    
    start_time = time.time()
    decompressor = AdaptiveCompressor(
        hidden_size=64,
        initial_learning_rate=0.001,
        min_learning_rate=0.00005,
        decay_rate=0.9999,
        encode_method=compression_method,
        batch_size=64
    )
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
