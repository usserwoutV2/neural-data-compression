from typing import List
import io
from arithmeticEncoder import SimpleFrequencyTable, ArithmeticEncoder, ArithmeticDecoder, BitOutputStream, BitInputStream
from huffman import HuffmanCoding
import lzma
import zlib
import bz2
import gzip
from LZ78Encoder import LZ78Encoder

class Encoder:
    
    def __init__(self, method:str = "lzma"):
        assert method in ["arithmetic", "huffman", "lzma", "zlib", "bz2", "gzip", "lz78"], "method must be 'arithmetic', 'huffman', 'lzma', 'zlib', 'bz2', 'lz78' or 'gzip'"        
        self.method = method   
            

    def _encode(self, input_indices: List[int], freq: List[float]) -> bytes:
        
        if self.method == "arithmetic":
            freqs = SimpleFrequencyTable(freq)
            output_stream = io.BytesIO()
            bit_output = BitOutputStream(output_stream)
            encoder = ArithmeticEncoder(32, bit_output)

            for symbol in input_indices:
                encoder.write(freqs, symbol)
            encoder.finish()

            return output_stream.getvalue()
        elif self.method == "huffman":
            huffman = HuffmanCoding()
            return huffman.compress(input_indices, freq)
        elif self.method == "lzma":
            data = bytes(input_indices)
            return lzma.compress(data)
        elif self.method == "zlib":
            data = bytes(input_indices)
            return zlib.compress(data)
        elif self.method == "bz2":
            data = bytes(input_indices)
            return bz2.compress(data)
        elif self.method == "gzip":
            data = bytes(input_indices)
            with io.BytesIO() as byte_stream:
                with gzip.GzipFile(fileobj=byte_stream, mode='wb') as gzip_file:
                    gzip_file.write(data)
                return byte_stream.getvalue()
        elif self.method == "lz78":
            compressor = LZ78Encoder()
            compressed_data = compressor.compress(input_indices)
            return compressed_data

    def _decode(self, encoded_data: bytes, freq: List[float], output_size: int) -> List[int]:
        if self.method == "arithmetic":
            freqs = SimpleFrequencyTable(freq)
            input_stream = io.BytesIO(encoded_data)
            bit_input = BitInputStream(input_stream)
            decoder = ArithmeticDecoder(32, bit_input)

            decoded_indices = []
            for _ in range(output_size):
                symbol = decoder.read(freqs)
                decoded_indices.append(symbol)

            return decoded_indices
        elif self.method == "huffman":
            huffman = HuffmanCoding()
            return huffman.decompress(encoded_data, freq)
        elif self.method == "lzma":
            data = lzma.decompress(encoded_data)
            return list(data)
        elif self.method == "zlib":
            data = zlib.decompress(encoded_data)
            return list(data)
        elif self.method == "bz2":
            data = bz2.decompress(encoded_data)
            return list(data)
        elif self.method == "gzip":
            with io.BytesIO(encoded_data) as byte_stream:
                with gzip.GzipFile(fileobj=byte_stream, mode='rb') as gzip_file:
                    data = gzip_file.read()
            return list(data)
        elif self.method == "lz78":
            compressor = LZ78Encoder()
            return compressor.decompress(encoded_data)
    
    
    def _encode_str(self, input_string:str):
        input_indices = [ord(c) for c in input_string]
        freq = [0] * 256
        for index in input_indices:
            freq[index] += 1
        return self._encode(input_indices, freq)


class AdaptiveEncoder:
    def __init__(self, alphabet_size: int, method: str = "arithmetic"):
        assert method in ["arithmetic", "gzip", "lz78"], "method must be 'arithmetic', 'gzip', or 'lz78'"
        self.alphabet_size = alphabet_size
        self.method = method
        self.freqs = SimpleFrequencyTable([1] * alphabet_size)
        self.encoder = None
        self.output_stream = None
        self.bit_output = None
        self.symbols = []  # Collect symbols for gzip and lz78

    def start_encoding(self):
        if self.method == "arithmetic":
            self.output_stream = io.BytesIO()
            self.bit_output = BitOutputStream(self.output_stream)
            self.encoder = ArithmeticEncoder(32, self.bit_output)
        elif self.method == "gzip":
            self.output_stream = io.BytesIO()
            self.encoder = gzip.GzipFile(fileobj=self.output_stream, mode='wb')
        elif self.method == "lz78":
            self.encoder = LZ78Encoder()
            self.encoder.start_encoding()

    def encode_symbol(self, symbol: int):
        if self.encoder is None:
            raise ValueError("Encoder not started. Call start_encoding() first.")
        if self.method == "arithmetic":
            self.encoder.write(self.freqs, symbol)
            self.freqs.increment(symbol)
        elif self.method == "gzip":
            self.encoder.write(bytes([symbol]))
        elif self.method == "lz78":
            self.encoder.encode_symbol(symbol)
            

    def finish_encoding(self) -> bytes:
        if self.encoder is None:
            raise ValueError("Encoder not started. Call start_encoding() first.")
        if self.method == "arithmetic":
            self.encoder.finish()
            return self.output_stream.getvalue()
        elif self.method == "gzip":
            self.encoder.close()
            return self.output_stream.getvalue()
        elif self.method == "lz78":
            return self.encoder.finish_encoding()

    def decode(self, encoded_data: bytes, output_size: int) -> List[int]:
        if self.method == "arithmetic":
            input_stream = io.BytesIO(encoded_data)
            bit_input = BitInputStream(input_stream)
            decoder = ArithmeticDecoder(32, bit_input)

            decoded_indices = []
            for _ in range(output_size):
                symbol = decoder.read(self.freqs)
                decoded_indices.append(symbol)
                self.freqs.increment(symbol)

            return decoded_indices
        elif self.method == "gzip":
            input_stream = io.BytesIO(encoded_data)
            with gzip.GzipFile(fileobj=input_stream, mode='rb') as gzip_file:
                data = gzip_file.read()
            return list(data)
        elif self.method == "lz78":
            decoder = LZ78Encoder()
            return decoder.decode(encoded_data, output_size)
        
        
    def get_next_best(self):
        if self.method == "lz78":
            return self.encoder.get_best_next_symbols()
        else:
            raise ValueError("Method must be 'lz78'")