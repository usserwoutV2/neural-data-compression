import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import sys
import os
from SupporterModel import SupporterModel
import pickle
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Encoder import Encoder
from exampleData import sample2, sample1, sample3, sample4
from stats import show_plot
from entropy import calculate_entropy_list, calculate_entropy
from util import set_seed, load_dataset
import numpy as np

import lzma
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import BertProcessing
# 274165) 172248 -> 145974 -> 128625
class DynamicCompressor(Encoder):
    def __init__(self, hidden_size: int = 64, learning_rate: float = 0.001, epochs: int = 10, encode_method="arithmetic"):
        super().__init__(method=encode_method)
        
        # Initialize the DynamicCompressor with hyperparameters
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()
        self.tokenizer.post_processor = BertProcessing(("[CLS]", 1), ("[SEP]", 2))
        self.vocab_size = 150 
        
    def train_tokenizer(self, texts: List[str]):
        self.vocab_size = max(150, min(1000, round(len(texts[0]) / 333)  ))
        self.trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=["[CLS]", "[SEP]"])
        
        self.tokenizer.train_from_iterator(texts, self.trainer)

    def _string_to_indices(self, input_string: str) -> List[int]:
        # Convert a string to a list of indices using the tokenizer
        return self.tokenizer.encode(input_string).ids

    def _indices_to_string(self, indices: List[int]) -> str:
        # Convert a list of indices back to a string using the tokenizer
        return self.tokenizer.decode(indices)

    def train(self, input_string: str):
        # Train the SupporterModel on the input string
        self.train_tokenizer([input_string])
        # Initialize the SupporterModel with the correct vocabulary size
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.supporter_model = SupporterModel(self.hidden_size, self.hidden_size, vocab_size=self.vocab_size, quantize=True)
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
        probabilities = torch.softmax(logits, dim=1) # TODO: can we remove this?
        compressed_indices = []
        for i, prob in enumerate(probabilities):
            target_index = input_indices[i + 1]
            target_prob = prob[target_index].item()

            # Count how many probabilities are higher than the target_prob
            rank = (prob > target_prob).sum().item()
            compressed_indices.append(rank)

            
        show_plot(compressed_indices)
        # print("Entropy before transformation",calculate_entropy(input_string))
        # print("Entropy after transformation",calculate_entropy_list(compressed_indices))

        # Calculate frequency of each index for arithmetic coding
        freq = [0] * self.vocab_size
        for index in compressed_indices:
            
            freq[index] += 1
        
        #print("compress indices:  ", compressed_indices)
        # Use arithmetic coding to further compress the data
        first_char_index = input_indices[0]      
        encoded_data = self._encode(compressed_indices, freq)
        
        return encoded_data, freq, first_char_index, len(compressed_indices)
    
    def decompress(self, compressed_data: bytes, freq: List[float], output_size: int, first_char_index: int) -> str:
        # Decompress the data using arithmetic decoding and the SupporterModel

        decoded_indices = self._decode(compressed_data, freq, output_size )
        max_seq_len = 20  # Define a fixed maximum sequence length
        decompressed_indices = [first_char_index]  # Initialize the list to hold decompressed indices
        input_buffer = [first_char_index]
        
        # Initialize the input buffer with zeros or a start token
        input_buffer = [0] * (max_seq_len - 1) + input_buffer
        input_tensor = torch.tensor([input_buffer], dtype=torch.long)

        supporter_model = self.supporter_model
        supporter_model.eval()  # Set the model to evaluation mode

        for i in range(output_size ):
            with torch.no_grad():
                # Get the model's output for the last position
                logits = supporter_model(input_tensor)[0, -1]

            probs = torch.softmax(logits, dim=0)
            if i >= len(decoded_indices):
                break
            target_index = decoded_indices[i]
            next_index = torch.topk(probs, target_index + 1).indices[-1].item()

            decompressed_indices.append(next_index)

            input_buffer.append(next_index)
            input_buffer = input_buffer[-max_seq_len:]
            input_tensor = torch.tensor([input_buffer], dtype=torch.long)
            
        # Convert indices back to a string
        return self._indices_to_string(decompressed_indices)[2:]
    
    def save_compressed_data(self, model_save_path: str, compressed_data: bytes, freq: List[float], first_char_index: int,indices_length:int):
        # Save the model's state dictionary and compressed data into one file
        data = {
            'model_state_dict': self.supporter_model.state_dict(),
            'compressed_data': compressed_data,
            'freq': freq,
            'tokenizer': lzma.compress(self.tokenizer.to_str().encode('utf-8')),
            'first_char_index': first_char_index,
            'indices_length':indices_length
        }
        
        print("Tokenized size", len(data['tokenizer']))
        print("Total size:", len(compressed_data) + len(data['tokenizer']))
        with open(model_save_path, 'wb') as f:
            pickle.dump(data, f)
            


    def load_compressed_data(self, model_save_path: str):
        # Load the model's state dictionary and compressed data from one file
        with open(model_save_path, 'rb') as f:
            data = pickle.load(f)
        first_char_index = data['first_char_index']
        self.tokenizer = Tokenizer.from_str(lzma.decompress(data['tokenizer']).decode('utf-8')) 
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.supporter_model = SupporterModel(self.hidden_size, self.hidden_size, self.vocab_size)
        self.supporter_model.load_state_dict(data['model_state_dict'])
        self.supporter_model.eval()

        compressed_data = data['compressed_data']
        freq = data['freq']
        
        return compressed_data, freq, first_char_index, data['indices_length']

encode_method = "arithmetic"
input_size = 50_000
input_string = load_dataset("bible", input_size)
# TODO:
# - Optimize stuff
# - Right now we save unnecessary data in the `save_compressed_data` function, like first_char_index. Find a better way.
def main():
    set_seed(421)
    print(f"Original data size: {len(input_string)} bytes")
    #show_plot(input_string)
    
    hidden_size = 58
    learning_rate = 0.0100963743367759346
    epochs = 20
    
    compressor = DynamicCompressor(hidden_size=hidden_size, epochs=epochs, learning_rate=learning_rate, encode_method=encode_method)
    
    start_time = time.time()
    compressed_data, freq, first_char_index, indices_length = compressor.compress(input_string)
    print(f"Compression time: {time.time() - start_time:.2f} seconds")
    
    compressor.save_compressed_data("compressed_data.pkl", compressed_data, freq, first_char_index,indices_length)
    print(f"Compressed data size: {len(compressed_data)} bytes")
    
    
    decompressor = DynamicCompressor(hidden_size=hidden_size, epochs=epochs, learning_rate=learning_rate, encode_method=encode_method)
    
    compressed_data, freq, first_char_index,indices_length  = decompressor.load_compressed_data("compressed_data.pkl")

    
    decompressed_string = decompressor.decompress(compressed_data, freq, indices_length, first_char_index)
    if input_string != decompressed_string:
        print("Strings do not match!")
        print(input_string[:180])
        print("-------------------------------------")
        print(decompressed_string[:180])
    else:
        print("Decompression successful!")
    
    
    
def compress_without_model():
    
    encoder = Encoder(method=encode_method)
    compressed = encoder._encode_str(input_string)
    print(f"Compressed data size (no support model): {len(compressed)} bytes")


if __name__ == "__main__":
    main()
    compress_without_model()