
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