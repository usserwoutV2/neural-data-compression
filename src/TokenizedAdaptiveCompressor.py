import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
from tqdm import tqdm  
import numpy as np
import time
import lzma

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import BertProcessing

from encoders.Encoder import Encoder, AdaptiveEncoder
from SupporterModel import SupporterModel
from util.stats import show_plot
from util.util import  load_dataset, set_seed
from util.match_string import match_string

del_me = []

class TokenizedAdaptiveCompressor(Encoder):
    def __init__(self, hidden_size: int = 64, seed=42, initial_learning_rate: float = 0.1, min_learning_rate: float = 0.001, decay_rate: float = 0.99, batch_size=8, encode_method="arithmetic", input_type="utf8"):
        set_seed(seed)
        self.hidden_size = hidden_size
        self.initial_learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.decay_rate = decay_rate
        self.current_step = 0
        self.vocab_size = 500
        self.supporter_model = None
        self.use_rnn= True
        self.sequence_length = 32
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.encode_method = encode_method
        
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()
        self.tokenizer.post_processor = BertProcessing(("[CLS]", 1), ("[SEP]", 2),)
        self.input_type = input_type

    def _create_vocabulary(self):

        self.supporter_model = SupporterModel(self.hidden_size, self.hidden_size, self.vocab_size , quantize=False)
        self.optimizer = optim.Adam(self.supporter_model.parameters(), lr=self.initial_learning_rate)

    def _update_learning_rate(self):
        new_lr = max(self.min_learning_rate, self.initial_learning_rate * (self.decay_rate ** (self.current_step )))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def _string_to_indices(self, input_string: str) -> List[int]:
        return self.tokenizer.encode(input_string).ids

    def _indices_to_string(self, indices: List[int]) -> str:
        return self.tokenizer.decode(indices)

    def train_tokenizer(self, texts: List[str]):
        self.trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=["[CLS]", "[SEP]"])
        
        self.tokenizer.train_from_iterator(texts, self.trainer)
    
    def _string_to_ord(self, input_string: str) -> List[int]:
        return [ord(char) for char in input_string]

    def _ord_to_string(self, indices: List[int]) -> str:
        return ''.join(chr(idx) for idx in indices)
    
    def compress(self, input_string: str) -> bytes:
        if isinstance(input_string, bytes):
            input_string = input_string.decode('latin1')

        self.vocab_size = max(150 if self.input_type == "utf8" else 260, min(1500, round(len(input_string) / 250)  ))
        self._create_vocabulary()
        self.train_tokenizer([input_string])
        
        input_indices = self._string_to_indices(input_string)

        num_samples = len(input_indices)
        compressed_indices = []

        adaptive_encoder = AdaptiveEncoder(self.vocab_size, method=self.encode_method)
        adaptive_encoder.start_encoding()

        # Include the initial sequence in the compressed data
        prefix = np.array(input_indices[:self.sequence_length], dtype=np.uint16).tobytes()


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
        global del_me
        del_me = compressed_indices
        input_size_bytes = len(compressed_indices).to_bytes(8, byteorder='big')
        
        tokens = lzma.compress(self.tokenizer.to_str().encode('utf-8'))
        tokens_size_bytes = len(tokens).to_bytes(8, byteorder='big')
        
        vocab_size_bytes = self.vocab_size.to_bytes(2, byteorder='big')
            
        return input_size_bytes + tokens_size_bytes + vocab_size_bytes + prefix + tokens + encoded_data


    def decompress(self, compressed_data: bytes) -> str: 
        num_symbols = int.from_bytes(compressed_data[:8], byteorder='big')
        tokens_size = int.from_bytes(compressed_data[8:16], byteorder='big')
        self.vocab_size = int.from_bytes(compressed_data[16:18], byteorder='big')
        self._create_vocabulary()
        prefix_indices = list(np.frombuffer(compressed_data[18:18 + self.sequence_length*2], dtype=np.uint16))
        
        tokens = compressed_data[18 + self.sequence_length*2:18 + self.sequence_length*2 + tokens_size]
        encoded_data = compressed_data[18 + self.sequence_length*2 + tokens_size:]
        
        self.tokenizer = Tokenizer.from_str(lzma.decompress(tokens).decode('utf-8')) 

        decompressed_indices = prefix_indices.copy()
        adaptive_decoder = AdaptiveEncoder(self.vocab_size, method=self.encode_method)
        decoded_ranks = adaptive_decoder.decode(encoded_data, num_symbols)
                
        batch_input_sequences = []
        batch_targets = []
        
        global del_me

        for i in tqdm(range(num_symbols), desc="Decompressing"):
            input_sequence = decompressed_indices[-self.sequence_length:]
            input_tensor = torch.tensor([input_sequence], dtype=torch.long)

            with torch.no_grad():
                output = self.supporter_model(input_tensor)
                probs = torch.softmax(output[:, -1, :], dim=1)
                target_index = torch.topk(probs, decoded_ranks[i] + 1, dim=1).indices[0, -1].item()
                decompressed_indices.append(target_index)
                
                # if target_index != del_me[i]:
                #     print(f"Index {i} does not match! {target_index} != {del_me[i]}")
                #     exit(1)
                    
                
        
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
            decompressed_string = decompressed_string.encode('latin1')
        if decompressed_string[0] != ' ': # remove special character from tokenizer
            return decompressed_string[2:] 
        return decompressed_string[1:]

compression_method = "arithmetic"
input_type = "utf8"
input_string = load_dataset("bible", 1_000_000)


def main():
    
    print(f"Original data size: {len(input_string)} bytes")
    #show_plot(input_string)
    
    start_time = time.time()

    compressor = TokenizedAdaptiveCompressor(hidden_size=64, initial_learning_rate=0.001, min_learning_rate=0.00005, decay_rate=0.9999, encode_method=compression_method, batch_size=128, input_type=input_type)
    compressed_data = compressor.compress(input_string)
    print(f"Compression took {time.time() - start_time:.2f} seconds")
    print(f"Compressed data size: {len(compressed_data)} bytes")

    start_time = time.time()
    decompressor = TokenizedAdaptiveCompressor(hidden_size=64, initial_learning_rate=0.001, min_learning_rate=0.00005, decay_rate=0.9999, encode_method=compression_method, batch_size=128, input_type=input_type)
    decompressed_string = decompressor.decompress(compressed_data)
    print(f"Decompression took {time.time() - start_time:.2f} seconds")

    
    if match_string(input_string, decompressed_string):
        print("Decompression successful!")

def compress_without_model():
    
    encoder = Encoder(method=compression_method)
    compressed = encoder._encode_str(input_string)
    print(f"Compressed data size (no support model): {len(compressed)} bytes")

if __name__ == "__main__":
    main()
    #compress_without_model()