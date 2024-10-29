import os
import sys
import io
import time
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from arithmeticEncoder import (
    SimpleFrequencyTable,
    ArithmeticEncoder,
    ArithmeticDecoder,
    BitOutputStream,
    BitInputStream,
)
from DNAModel import DNAModel
from exampleData import sample1, sample2
from stats import calculate_frequencies
from Encoder import Encoder


class StaticCompressor:
    def __init__(self, model: DNAModel, alphabet: str = "ACGT"):
        self.model = model
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.freq = np.zeros(self.alphabet_size, dtype=int)

    def compress(self, input_string: str) -> bytes:
        indices = self._translate_to_index(input_string)
        self._calculate_frequencies(indices)
        calculate_frequencies(indices)
        return self._arithmetic_encode(indices)

    def decompress(self, compressed_data: bytes, output_size: int, first_n_characters: str) -> str:
        decoded_indices = self._arithmetic_decode(compressed_data, output_size)
        return self._translate_from_index(decoded_indices, first_n_characters)

    def _translate_to_index(self, input_string: str) -> np.ndarray:
        output = np.zeros(len(input_string) - 1, dtype=int)
        model_input_length = 1
        total_iterations = len(input_string) - model_input_length - 1

        for i in range(model_input_length, len(input_string) - 1):
            substr = input_string[i-19:i+1]
            predicted = self.model.predict_next_chars(substr)
            #print(f"After '{substr}', predicted next chars: {predicted} Expected: {input_string[i + 1].lower()}")
            
            if len(substr) != model_input_length + 1:
                continue
            
            predicted = self.model.predict_next_chars(substr)
            try:
                index = predicted.index(input_string[i + 1].upper())
            except ValueError:
                index = 0  # Default to the first index if the character isn't in predictions

            output[i - model_input_length] = index
            self.freq[index] += 1
            original_counts[input_string[i + 1].upper()] += 1

            if input_string[i + 1].lower() == predicted[0].lower():
                correct_predictions += 1

        accuracy = correct_predictions / (len(input_string) - 1) * 100
        print(f"Accuracy: {accuracy:.2f}%")
        print("Stats:", self.freq)
        print("Original counts:", original_counts)

        return output

    def _translate_from_index(self, input_indices: np.ndarray, first_n_characters: str) -> str:
        output = list(first_n_characters)
        model_input_length = len(first_n_characters)

        for i in range(len(input_indices)):
            substr = "".join(output[model_input_length + i - 20 : model_input_length + i + 1])
            predicted = self.model.predict_next_chars(substr)
            #print(f"After '{substr}', predicted next chars: {predicted}")
            output.append(predicted[input_indices[i]])

        return "".join(output)

    def _calculate_frequencies(self, indices: np.ndarray) -> None:
        # Efficient frequency calculation using numpy
        unique, counts = np.unique(indices, return_counts=True)
        for idx, count in zip(unique, counts):
            self.freq[idx] = count

    def _arithmetic_encode(self, input_indices: np.ndarray) -> bytes:
        freqs = SimpleFrequencyTable(self.freq.tolist())
        input_stream = io.BytesIO()
        bit_output = BitOutputStream(input_stream)
        encoder = ArithmeticEncoder(32, bit_output)

        print(f"Input: {input_indices}")
        for number in input_indices:
            encoder.write(freqs, int(number))
        encoder.finish()

        encoded_data = input_stream.getvalue()
        bit_output.close()

        compression_ratio = len(encoded_data) / len(input_indices)
        print(f"Size of encoded data: {len(encoded_data)} bytes, original text: {len(input_indices)} bytes, compression ratio: {compression_ratio:.2f}")
        return encoded_data

    def _arithmetic_decode(self, encoded_data: bytes, output_size: int) -> np.ndarray:
        freqs = SimpleFrequencyTable(self.freq.tolist())
        input_stream = io.BytesIO(encoded_data)
        bit_input = BitInputStream(input_stream)
        decoder = ArithmeticDecoder(32, bit_input)

        decoded_list = np.zeros(output_size - 20, dtype=int)
        for i in range(output_size - 20):
            decoded_list[i] = decoder.read(freqs)

        return decoded_list


def load_dataset(filename):
    with open(filename, "r") as file:
        return file.read()

def main():
    dataset_path = 'datasets/files_to_be_compressed/celegchr_ultrasmall.txt'
    with open(dataset_path, 'r') as file:
        input_string = file.read()
    model = DNAModel(input_string)
    compressor = StaticCompressor(model)
    start_time = time.time()
    compressed_data = compressor.compress(input_string)
    end_time = time.time()
    print(f"Compression time: {end_time - start_time:.2f} seconds")
    print(f"Compressed size: {len(compressed_data)} bytes")
    print(f"Original size: {len(input_string)} bytes")

    # There is still an issue with the decompression, sometimes the ending is not correct
    # This issue has something to do with arithmetic encoding/decoding
    # decompressed_str = compressor.decompress(
    #     compressed_data, len(input_string), input_string[:20]
    # )
    # print(f"Decompressed string: {decompressed_str}")
    # print(f"Original string:     {input_string}")
    # print(f"Strings match: {decompressed_str == input_string}")

def compress_without_model():
    input_string = sample2
    print(f"Original data size: {len(input_string)} bytes")
    
    encoder = Encoder()
    compressed = encoder._arithmetic_encode_str(input_string)
    print(f"Compressed data size (no support model): {len(compressed)} bytes")



if __name__ == "__main__":
    main(show_graphs=False)
    compress_without_model()
    
