from typing import List
import io
from arithmeticEncoder import SimpleFrequencyTable, ArithmeticEncoder, ArithmeticDecoder, BitOutputStream, BitInputStream



class Encoder:
  def _arithmetic_encode(self, input_indices: List[int], freq: List[float]) -> bytes:
      freqs = SimpleFrequencyTable(freq)
      output_stream = io.BytesIO()
      bit_output = BitOutputStream(output_stream)
      encoder = ArithmeticEncoder(32, bit_output)

      for symbol in input_indices:
          encoder.write(freqs, symbol)
      encoder.finish()

      return output_stream.getvalue()

  def _arithmetic_decode(self, encoded_data: bytes, freq: List[float], output_size: int) -> List[int]:
      freqs = SimpleFrequencyTable(freq)
      input_stream = io.BytesIO(encoded_data)
      bit_input = BitInputStream(input_stream)
      decoder = ArithmeticDecoder(32, bit_input)

      decoded_indices = []
      for _ in range(output_size):
          symbol = decoder.read(freqs)
          decoded_indices.append(symbol)

      return decoded_indices
  
  
  
  
  def _arithmetic_encode_str(self, input_string:str):
    input_indices = [ord(c) for c in input_string]
    freq = [0] * 256
    for index in input_indices:
        freq[index] += 1
    
    encoder = Encoder()
    compressed = encoder._arithmetic_encode(input_indices, freq)
    return compressed


class AdaptiveArithmeticEncoder:
    def __init__(self, alphabet_size: int):
        self.alphabet_size = alphabet_size
        self.freqs = SimpleFrequencyTable([1] * alphabet_size)
        self.encoder = None
        self.output_stream = None
        self.bit_output = None

    def start_encoding(self):
        self.output_stream = io.BytesIO()
        self.bit_output = BitOutputStream(self.output_stream)
        self.encoder = ArithmeticEncoder(32, self.bit_output)

    def encode_symbol(self, symbol: int):
        if self.encoder is None:
            raise ValueError("Encoder not started. Call start_encoding() first.")
        self.encoder.write(self.freqs, symbol)
        self.freqs.increment(symbol)

    def finish_encoding(self) -> bytes:
        if self.encoder is None:
            raise ValueError("Encoder not started. Call start_encoding() first.")
        self.encoder.finish()
        return self.output_stream.getvalue()

    def decode(self, encoded_data: bytes, output_size: int) -> List[int]:
        input_stream = io.BytesIO(encoded_data)
        bit_input = BitInputStream(input_stream)
        decoder = ArithmeticDecoder(32, bit_input)

        decoded_indices = []
        for _ in range(output_size):
            symbol = decoder.read(self.freqs)
            decoded_indices.append(symbol)
            self.freqs.increment(symbol)

        return decoded_indices


