import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Union
import os
from SupporterModel import SupporterModel
import pickle
import time
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from tqdm import tqdm  


from encoders.Encoder import Encoder
from util.util import set_seed, load_dataset
from util.stats import show_plot

from util.match_string import match_string

import lzma
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import BertProcessing



class DynamicCompressor(Encoder):
    def __init__(self, hidden_size: int = 64, learning_rate: float = 0.001, epochs: int = 10, encode_method="arithmetic", alphabet_size=128, input_type="utf8"):
        super().__init__(method=encode_method)
        
        # Initialize the DynamicCompressor with hyperparameters
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()
        self.tokenizer.post_processor = BertProcessing(("[CLS]", 1), ("[SEP]", 2))
        self.vocab_size = 0 
        self.use_rnn=False
        self.alphabet_size = alphabet_size
        self.input_type = input_type
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
    def train_tokenizer(self, texts: List[str]):
        self.vocab_size = round(max(150, min(1000, round(len(texts[0]) / 200)  ))  * (self.alphabet_size / 128))
        print(f"Vocab size: {self.vocab_size}")
        self.trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=["[CLS]", "[SEP]"])
        
        self.tokenizer.train_from_iterator(texts, self.trainer)

    def _string_to_indices(self, input_string: str) -> List[int]:
        return self.tokenizer.encode(input_string).ids

    def _indices_to_string(self, indices: List[int]) -> str:
        return self.tokenizer.decode(indices)
    
    def train(self, input_string: Union[str, bytes]):
        if isinstance(input_string, bytes):
            input_string = input_string.decode('latin1')
        
        self.train_tokenizer([input_string])
        
        # Initialize the SupporterModel with the correct vocabulary size
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.supporter_model = SupporterModel(self.hidden_size, self.hidden_size, vocab_size=self.vocab_size, quantize=True, use_rnn=self.use_rnn)
        self.optimizer = optim.Adam(self.supporter_model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        
        # Prepare input and target tensors
        input_indices = self._string_to_indices(input_string)
        max_length =  50_000
        
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
        
        learning_time = time.time()
        input_indices = self.train(input_string)
        print(f"Learning time: {time.time() - learning_time:.2f} seconds")
        
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
        
        # Calculate frequency of each index for arithmetic coding
        freq = [0] * self.vocab_size
        for index in compressed_indices:
            freq[index] += 1
        
        # Use arithmetic coding to further compress the data
        first_char_index = input_indices[0]
        show_plot(compressed_indices)
        encoded_data = self._encode(compressed_indices, freq, 1 if self.vocab_size < 256 else 2)

        return encoded_data, freq, first_char_index, len(compressed_indices)
    
    def decompress(self, compressed_data: bytes, freq: List[float], output_size: int, first_char_index: int) -> Union[str, bytes]:
        # Decompress the data using arithmetic decoding and the SupporterModel

        decoded_indices = self._decode(compressed_data, freq, output_size, bytes_per_element=1 if self.vocab_size < 256 else 2)
        max_seq_len = 64  if self.use_rnn else 10
        decompressed_indices = [first_char_index]  # Initialize the list to hold decompressed indices
        input_buffer = [first_char_index]
        
        # Initialize the input buffer with zeros or a start token
        #input_buffer = [0] * (max_seq_len - 1) + input_buffer
        input_tensor = torch.tensor([input_buffer], dtype=torch.long)

        supporter_model = self.supporter_model
        supporter_model.eval()  # Set the model to evaluation mode

        for i in tqdm(range(output_size), desc="Decompressing"):
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
        decompressed_string = self._indices_to_string(decompressed_indices)
        if decompressed_string[0] == ' ':
            decompressed_string = decompressed_string[1:]
        else: 
            decompressed_string = decompressed_string[2:]
        
        if self.input_type == "bytes":
            return decompressed_string.encode('latin1')
        else:
            return decompressed_string
    
    def save_compressed_data(self, model_save_path: str, compressed_data: bytes, freq: List[float], first_char_index: int,indices_length:int):
        # Save the model's state dictionary and compressed data into one file
        data = {
            'model_state_dict': self.supporter_model.state_dict(),
            'compressed_data': compressed_data,
            'freq': freq,
            'tokenizer': lzma.compress(self.tokenizer.to_str().encode('utf-8')),
            'first_char_index': first_char_index,
            'indices_length': indices_length
        }
        
        print("Tokenized size", len(data['tokenizer']))
        print("===> Total size:", len(compressed_data) + len(data['tokenizer']))
        print("Model size:", len(pickle.dumps(data['model_state_dict'])))
        with open(model_save_path, 'wb') as f:
            pickle.dump(data, f)
            file_size = f.tell()
        
        print(f"Total bytes written to file: {file_size}")
            


    def load_compressed_data(self, model_save_path: str):
        # Load the model's state dictionary and compressed data from one file
        with open(model_save_path, 'rb') as f:
            data = pickle.load(f)
        first_char_index = data['first_char_index']
        self.tokenizer = Tokenizer.from_str(lzma.decompress(data['tokenizer']).decode('utf-8')) 
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.supporter_model = SupporterModel(self.hidden_size, self.hidden_size, self.vocab_size, quantize=True, use_rnn=self.use_rnn)
        self.supporter_model.load_state_dict(data['model_state_dict'])
        self.supporter_model.eval()

        compressed_data = data['compressed_data']
        freq = data['freq']
        
        return compressed_data, freq, first_char_index, data['indices_length']
encode_method = "arithmetic"
input_type = "utf8"
input_string = load_dataset("bible", 1_000_000)


def main():
    set_seed(421)
    print(f"Original data size: {len(input_string)} bytes")
    show_plot(input_string)
    
    hidden_size = 58
    learning_rate = 0.01
    epochs = 50
    
    compressor = DynamicCompressor(hidden_size=hidden_size, epochs=epochs, learning_rate=learning_rate, encode_method=encode_method, input_type=input_type)
    
    start_time = time.time()
    
    compressed_data, freq, first_char_index, indices_length = compressor.compress(input_string)
    print(f"Total compression time: {time.time() - start_time:.2f} seconds")
    
    compressor.save_compressed_data("compressed_data.pkl", compressed_data, freq, first_char_index,indices_length)
    print(f"Compressed data size: {len(compressed_data)} bytes")
    
    decompressor = DynamicCompressor(hidden_size=hidden_size, epochs=epochs, learning_rate=learning_rate, encode_method=encode_method, input_type=input_type)
    
    compressed_data, freq, first_char_index,indices_length  = decompressor.load_compressed_data("compressed_data.pkl")

    
    start_time = time.time()
    decompressed_string = decompressor.decompress(compressed_data, freq, indices_length, first_char_index)
    print(f"Decompression time: {time.time() - start_time:.2f} seconds")
    if match_string(input_string, decompressed_string):
        print("Decompression successful!")
    
    
    
def compress_without_model():
    
    encoder = Encoder(method=encode_method)
    compressed = encoder._encode_str(input_string)
    print(f"Compressed data size (no support model): {len(compressed)} bytes")


if __name__ == "__main__":
    main()
    compress_without_model()