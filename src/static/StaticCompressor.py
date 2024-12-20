import os
import sys
import io
from typing import List
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'encoders')))
from arithmeticEncoder import SimpleFrequencyTable, ArithmeticEncoder, ArithmeticDecoder, BitOutputStream, BitInputStream
from DNAModel import DNAModel
from EnglishTextModel import EnglishTextModel
from StaticModel import StaticModel

class StaticCompressor:
    def __init__(self, model: StaticModel):
        self.model = model
        self.alphabet = model.legal_characters
        self.alphabet_size = len(self.alphabet)
        print("Alphabet size:", self.alphabet_size)
        self.freq = np.zeros(self.alphabet_size, dtype=int)
        self.char_to_index = {char: idx for idx, char in enumerate(self.alphabet)}
        self.index_to_char = {idx: char for idx, char in enumerate(self.alphabet)}

    def compress(self, input_string: str) -> bytes:
        indices = self._translate_to_index(input_string)
        return self._arithmetic_encode(indices)

    def decompress(self, compressed_data: bytes, output_size: int, first_n_characters: str) -> str:
        decoded_indices = self._arithmetic_decode(compressed_data, output_size)
        #print("Decoded:", decoded_indices)
        return self._translate_from_index(decoded_indices, first_n_characters)

    def _translate_to_index(self, input_string: str) -> List[int]:
        output = []
        original = {char: 0 for char in self.alphabet}
        correct = 0
        model_input_length = 19
        batch_size = 1024  # Adjust batch size based on your system's memory capacity

        # Prepare input tensor for batch predictions
        input_indices = [self.char_to_index.get(char, 0) for char in input_string]
        input_tensor = np.array([input_indices[i-19:i+1] for i in range(model_input_length, len(input_string) - 1)])

        # Process in batches
        num_batches = (len(input_tensor) + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(input_tensor))
            batch_input_tensor = input_tensor[batch_start:batch_end]

            # Use the model to predict probabilities
            predictions = self.model.predict(batch_input_tensor)

            for i, predicted_probs in enumerate(predictions):
                predicted_indices = np.argsort(predicted_probs)[-self.alphabet_size:][::-1]
                predicted_chars = [self.index_to_char[idx] for idx in predicted_indices]

                next_char = input_string[batch_start + i + model_input_length + 1]
                if next_char in predicted_chars:
                    index = predicted_chars.index(next_char)
                else:
                    index = self.char_to_index.get(next_char, 99)
                    print(f"Character '{next_char}' not in prediction. Using fallback index {index}.")

                output.append(index)
                self.freq[index] += 1
                if next_char in original:
                    original[next_char] += 1
                else:
                    original[next_char] = 1
                    print(f"Character '{next_char}' not found in original dictionary. Adding it with count 1.")

                if next_char == predicted_chars[0]:
                    correct += 1

        accuracy = correct / (len(input_string) - 1) * 100
        print(f"Accuracy: {accuracy:.2f}%")
        print("Stats:", self.freq)

        return output

    def _translate_from_index(self, input_indices: List[int], first_n_characters: str) -> str:
        output = list(first_n_characters)
        model_input_length = len(first_n_characters)
            
        for i in range(len(input_indices)):
            substr = "".join(output[model_input_length+i-20:model_input_length+i+1])
            input_indices_substr = [self.char_to_index.get(char, 0) for char in substr]
            input_tensor = np.array(input_indices_substr).reshape(1, -1)
            
            # Use the model to predict probabilities
            predictions = self.model.predict(input_tensor)[0]
            predicted_indices = np.argsort(predictions)[-self.alphabet_size:][::-1]
            predicted_chars = [self.index_to_char[idx] for idx in predicted_indices]
            
            output.append(predicted_chars[input_indices[i]])

        return "".join(output)

    def _arithmetic_encode(self, input_indices: List[int]) -> bytes:
        freqs = SimpleFrequencyTable(self.freq)
        input_stream = io.BytesIO()
        bit_output = BitOutputStream(input_stream)
        encoder = ArithmeticEncoder(32, bit_output)

        #print(f"Input:   {input_indices}")
        for number in input_indices:
            encoder.write(freqs, number)
        encoder.finish()

        encoded_data = input_stream.getvalue()
        bit_output.close()

        compression_ratio = len(encoded_data) / len(input_indices)
        print(f"Size of encoded data: {len(encoded_data)} bytes, original text: {len(input_indices)} bytes, compression ratio: {compression_ratio:.2f}")
        return encoded_data

    def _arithmetic_decode(self, encoded_data: bytes, output_size: int) -> List[int]:
        freqs = SimpleFrequencyTable(self.freq)
        input_stream = io.BytesIO(encoded_data)
        bit_input = BitInputStream(input_stream)
        decoder = ArithmeticDecoder(32, bit_input)

        decoded_list = []
        for _ in range(output_size - 20):
            symbol = decoder.read(freqs)
            decoded_list.append(symbol)

        return decoded_list

def main(dataset_path: str, model_type: str = 'english'):
    dataset_path = os.path.join(os.environ['VSC_HOME'], dataset_path)
    with open(dataset_path, 'r') as file:
        input_string = file.read()
    if model_type == 'dna':
        model = DNAModel(input_string)
    else:
        model = EnglishTextModel(input_string)
    compressor = StaticCompressor(model)

    start_time = time.time()
    compressed_data = compressor.compress(input_string)
    end_time = time.time()
    print(f"Compression time: {end_time - start_time:.2f} seconds")
    #print("Compressed data:", compressed_data)

    start_time = time.time()
    decompressed_str = compressor.decompress(compressed_data, len(input_string), input_string[:20])
    end_time = time.time()
    print(f"Decompression time: {end_time - start_time:.2f} seconds")
    #print(f"Decompressed string: {decompressed_str}")
    #print(f"Original string:     {input_string}")
    print(f"Strings match: {decompressed_str == input_string}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python StaticCompressor.py <dataset_path> [model_type]")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else 'english'
    main(dataset_path, model_type)