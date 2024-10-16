import os
import sys
import io
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from arithmeticEncoder import SimpleFrequencyTable, ArithmeticEncoder, ArithmeticDecoder, BitOutputStream, BitInputStream
from DNAModel import DNAModel

class StaticCompressor:
    def __init__(self, model: DNAModel, alphabet: str = 'ACGT'):
        self.model = model
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.freq = [0] * self.alphabet_size

    def compress(self, input_string: str) -> bytes:
        indices = self._translate_to_index(input_string)
        return self._arithmetic_encode(indices)

    def decompress(self, compressed_data: bytes, output_size: int, first_n_characters: str) -> str:
        decoded_indices = self._arithmetic_decode(compressed_data, output_size)
        print("Decoded:", decoded_indices)
        return self._translate_from_index(decoded_indices, first_n_characters)

    def _translate_to_index(self, input_string: str) -> List[int]:
        output = []
        
        original = {char: 0 for char in self.alphabet}
        correct = 0
        model_input_length = 19

        for i in range(model_input_length, len(input_string) - 1):
            substr = input_string[i-19:i+1]
            predicted = self.model.predict_next_chars(substr)
            print(f"After '{substr}', predicted next chars: {predicted} Expected: {input_string[i + 1].lower()}")
            
            index = predicted.index(input_string[i + 1].upper())
            output.append(index)
            self.freq[index] += 1
            original[input_string[i + 1].upper()] += 1
            if input_string[i + 1].lower() == predicted[0].lower():
                correct += 1

        accuracy = correct / (len(input_string) - 1) * 100
        print(f"Accuracy: {accuracy:.2f}%")
        print("Stats:", self.freq)
        print("Original:", original)
        
        return output

    def _translate_from_index(self, input_indices: List[int], first_n_characters: str) -> str:
        output = list(first_n_characters)
        model_input_length = len(first_n_characters)

        for i in range(len(input_indices)):
            substr = "".join(output[model_input_length+i-20:model_input_length+i+1])
            predicted = self.model.predict_next_chars(substr)
            print(f"After '{substr}', predicted next chars: {predicted}")
            output.append(predicted[input_indices[i]])

        return "".join(output)

    def _arithmetic_encode(self, input_indices: List[int]) -> bytes:
        freqs = SimpleFrequencyTable(self.freq)
        input_stream = io.BytesIO()
        bit_output = BitOutputStream(input_stream)
        encoder = ArithmeticEncoder(32, bit_output)

        print(f"Input:   {input_indices}")
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

def main():
    input_string = "AAGAAGATAGGCACTTTGTTACCCAAAAAACCACCCCTGAGT"
    model = DNAModel(input_string)
    compressor = StaticCompressor(model)

    compressed_data = compressor.compress(input_string)
    print("Compressed data:", compressed_data)

    # There is still an issue with the decompression, sometimes the ending is not correct
    # This issue has something to do with arithmetic encoding/decoding
    decompressed_str = compressor.decompress(compressed_data, len(input_string), input_string[:20])
    print(f"Decompressed string: {decompressed_str}")
    print(f"Original string:     {input_string}")
    print(f"Strings match: {decompressed_str == input_string}")

if __name__ == "__main__":
    main()