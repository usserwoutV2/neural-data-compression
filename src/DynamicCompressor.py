import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Union
import sys
import os
from  SupporterModel import SupporterModel
import pickle
import time
from tqdm import tqdm  


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from encoders.Encoder import Encoder
from util.stats import show_plot
from util.util import set_seed, load_dataset


class DynamicCompressor(Encoder):
    def __init__(self, hidden_size: int = 64, learning_rate: float = 0.01, epochs: int = 40, encode_method="arithmetic", input_type="utf8"):
        super().__init__(method=encode_method)
        
        # Initialize the DynamicCompressor with hyperparameters
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.use_rnn = False
        self.input_type = input_type

    

    def train(self, input_string: str):
        # Train the SupporterModel on the input string
        self._create_vocabulary(input_string)
        # Initialize the SupporterModel with the correct vocabulary size
        self.supporter_model = SupporterModel(self.hidden_size, self.hidden_size, vocab_size=self.vocab_size, quantize=True, use_rnn=self.use_rnn)
        self.optimizer = optim.Adam(self.supporter_model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Prepare input and target tensors
        input_indices = self._string_to_indices(input_string)
        max_length = 75_000 if self.use_rnn else 150_000
        
        if len(input_indices) <= max_length:
            input_tensor = torch.tensor(input_indices[:-1], dtype=torch.long).unsqueeze(0)
            target_tensor = torch.tensor(input_indices[1:], dtype=torch.long)
        else:
            input_tensor = torch.tensor(input_indices[:max_length], dtype=torch.long).unsqueeze(0)
            target_tensor = torch.tensor(input_indices[1:max_length+1], dtype=torch.long)
        
        print("Start training")
        # Training loop: update model parameters to minimize loss
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            output = self.supporter_model(input_tensor)
            loss = self.criterion(output.squeeze(0), target_tensor)
            loss.backward()

            self.optimizer.step()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
            
            # Calculate accuracy
        # with torch.no_grad():
        #     predictions = torch.argmax(output, dim=2).squeeze(0)
        #     correct_predictions = (predictions == target_tensor).sum().item()
        #     accuracy = correct_predictions / target_tensor.size(0)
        #     print(f"Final accuracy: {accuracy:.4f}")
        return input_indices

    def compress(self, input_data: Union[str, bytes]) -> Tuple[bytes, List[float], int, int]:
        # Compress the input string using the trained SupporterModel
        if isinstance(input_data, bytes):
            input_string = input_data.decode('latin1')
        else:
            input_string = input_data

        input_indices = self.train(input_string)
        
        # Prepare input tensor
        input_indices = self._string_to_indices(input_string)
        
        input_tensor = torch.tensor(input_indices[:-1], dtype=torch.long).unsqueeze(0)
        # Use the SupporterModel to predict probabilities
        with torch.no_grad():
            logits = self.supporter_model(input_tensor).squeeze(0)
        
        # Calculate probabilities and compress based on predictions
        probabilities = torch.softmax(logits, dim=1)
        
        # Find the ranks of the target indices
        sorted_probs, sorted_indices = torch.sort(probabilities, dim=1, descending=True)
        batch_inputs_tensor = torch.tensor(input_indices[1:], dtype=torch.long)
        mask = sorted_indices == batch_inputs_tensor.unsqueeze(1)
        
        # Find the rank positions
        ranks = torch.argmax(mask.int(), dim=1).tolist()
        
        compressed_indices = ranks
        show_plot(compressed_indices)

        # Calculate frequency of each index for arithmetic coding
        freq = [0] * self.vocab_size
        for index in compressed_indices:
            freq[index] += 1
        
        # Use arithmetic coding to further compress the data
        first_char_index = input_indices[0]
        encoded_data = self._encode(compressed_indices, freq)
        return encoded_data, freq, first_char_index
    
    def decompress(self, compressed_data: bytes, freq: List[float], output_size: int, first_char_index: int) -> Union[str, bytes]:
        # Decompress the data using arithmetic decoding and the SupporterModel
        decoded_indices = self._decode(compressed_data, freq, output_size - 1)

        max_seq_len = 64  if self.use_rnn else 10
        decompressed_indices = [first_char_index]  # Initialize the list to hold decompressed indices
        input_buffer = [first_char_index]
        
        # Initialize the input buffer with zeros or a start token
        input_buffer = [0] * (max_seq_len - 1) + input_buffer
        input_tensor = torch.tensor([input_buffer], dtype=torch.long)

        supporter_model = self.supporter_model
        supporter_model.eval()  # Set the model to evaluation mode

        for i in tqdm(range(output_size - 1), desc="Decompressing"):
            with torch.no_grad():
                # Get the model's output for the last position
                logits = supporter_model(input_tensor)[0, -1]

            probs = torch.softmax(logits, dim=0)
            target_index = decoded_indices[i]
            next_index = torch.topk(probs, target_index + 1).indices[-1].item()

            decompressed_indices.append(next_index)

            input_buffer.append(next_index)
            input_buffer = input_buffer[-max_seq_len:]
            input_tensor = torch.tensor([input_buffer], dtype=torch.long)

        # Convert indices back to a string
        decompressed_string = self._indices_to_string(decompressed_indices)
        
        if self.input_type == "bytes":
            return decompressed_string.encode('latin1')
        else:
            return decompressed_string
    
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
            file_size = f.tell()
        
        print(f"Total bytes written to file: {file_size}")
            


    def load_compressed_data(self, model_save_path: str):
        # Load the model's state dictionary and compressed data from one file
        with open(model_save_path, 'rb') as f:
            data = pickle.load(f)
        first_char_index = data['first_char_index']
        self.char_to_index = data['char_to_index']
        self.index_to_char = {idx: char for char, idx in self.char_to_index.items()}
        self.vocab_size = len(self.char_to_index)
        self.supporter_model = SupporterModel(self.hidden_size, self.hidden_size, self.vocab_size, use_rnn=self.use_rnn)
        self.supporter_model.load_state_dict(data['model_state_dict'])
        self.supporter_model.eval()
        
        compressed_data = data['compressed_data']
        freq = data['freq']
        
        return compressed_data, freq, first_char_index

encode_method = "huffman"
input_type = "utf8"
input_string = load_dataset("chr20", 1_000_000)


def main():
    set_seed(421)
    print(f"Original data size: {len(input_string)} bytes")
    show_plot(input_string)
    
    hidden_size = 58
    learning_rate = 0.0100963743367759346
    epochs = 50
    
    compressor = DynamicCompressor(hidden_size=hidden_size, epochs=epochs, learning_rate=learning_rate, encode_method=encode_method,input_type=input_type)
    
    start_time = time.time()
    compressed_data, freq, first_char_index = compressor.compress(input_string)
    print(f"Compression time: {time.time() - start_time:.2f} seconds")
    
    compressor.save_compressed_data("compressed_data.pkl", compressed_data, freq, first_char_index)
    print(f"Compressed data size: {len(compressed_data)} bytes")
    
    decompressor = DynamicCompressor(hidden_size=hidden_size, epochs=epochs, learning_rate=learning_rate, encode_method=encode_method,input_type=input_type)
    
    compressed_data, freq,first_char_index  = decompressor.load_compressed_data("compressed_data.pkl")

    
    start_time = time.time()
    decompressed_string = decompressor.decompress(compressed_data, freq, len(input_string), first_char_index)
    print(f"Decompression time: {time.time() - start_time:.2f} seconds")
    if input_string != decompressed_string:
        print("Strings do not match!")
        print(input_string[:100])
        print("-------------------------------------")
        print(decompressed_string[:100])
    else:
        print("Decompression successful!")
    
    
    
def compress_without_model():
    
    encoder = Encoder(method=encode_method)
    compressed = encoder._encode_str(input_string)
    print(f"Compressed data size (no support model): {len(compressed)} bytes")


if __name__ == "__main__":
    main()
    compress_without_model()
