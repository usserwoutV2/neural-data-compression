import io
from math import ceil, log2
from typing import List
class LZ78Encoder:
    def __init__(self):
        self.current_index = 1
        self.dictionary = {}
        self.w = []
        self.compressed_bits = []
        self.max_index = 0
        self.max_symbol = 0

    def _get_min_bits(self, max_value):
        return max(1, ceil(log2(max_value + 1)))

    def compress(self, input_data: List[int]) -> bytes:
        self.dictionary = {}
        self.w = []
        self.compressed_bits = []
        self.max_index = 0
        self.max_symbol = max(input_data) if input_data else 0

        for symbol in input_data:
            self._process_symbol(symbol)

        if self.w:
            index = self.dictionary.get(tuple(self.w), 0)
            self.compressed_bits.append((index, 0))
            self.max_index = max(self.max_index, index)

        return self._finalize_compression()

    def _process_symbol(self, symbol: int):
        ws = tuple(self.w + [symbol])
        if ws in self.dictionary:
            self.w.append(symbol)
        else:
            index = self.dictionary.get(tuple(self.w), 0)
            self.compressed_bits.append((index, symbol))
            self.dictionary[ws] = self.current_index
            self.current_index += 1
            self.w = []
            self.max_index = max(self.max_index, index)

    def _finalize_compression(self) -> bytes:
        index_bits = self._get_min_bits(self.max_index)
        symbol_bits = self._get_min_bits(self.max_symbol)

        bit_stream = 0
        bit_length = 0
        bytes_out = bytearray()

        for index, symbol in self.compressed_bits:
            bit_stream = (bit_stream << index_bits) | index
            bit_length += index_bits
            bit_stream = (bit_stream << symbol_bits) | symbol
            bit_length += symbol_bits
            while bit_length >= 8:
                bit_length -= 8
                byte = (bit_stream >> bit_length) & 0xFF
                bytes_out.append(byte)

        if bit_length > 0:
            byte = (bit_stream << (8 - bit_length)) & 0xFF
            bytes_out.append(byte)

        header = index_bits.to_bytes(1, 'big') + symbol_bits.to_bytes(1, 'big')
        return header + bytes(bytes_out)

    def decompress(self, compressed_data: bytes) -> List[int]:
        index_bits = compressed_data[0]
        symbol_bits = compressed_data[1]
        bit_length = len(compressed_data[2:]) * 8
        bit_stream = int.from_bytes(compressed_data[2:], 'big')
        dictionary = {}
        decompressed_data = []
        self.current_index = 1

        while bit_length >= (index_bits + symbol_bits):
            bit_length -= index_bits
            index = (bit_stream >> bit_length) & ((1 << index_bits) - 1)
            bit_length -= symbol_bits
            symbol = (bit_stream >> bit_length) & ((1 << symbol_bits) - 1)
            if index == 0:
                entry = [symbol]
            else:
                entry = dictionary[index] + [symbol]
            decompressed_data.extend(entry)
            dictionary[self.current_index] = entry
            self.current_index += 1

        return decompressed_data

    def get_dictionary(self):
        return self.dictionary
      
    def start_encoding(self):
        self.dictionary = {}
        self.w = []
        self.compressed_bits = []
        self.max_index = 0
        self.max_symbol = 0

    def encode_symbol(self, symbol: int):
        self._process_symbol(symbol)
        self.max_symbol = max(self.max_symbol, symbol)

    def finish_encoding(self) -> bytes:
        if self.w:
            index = self.dictionary.get(tuple(self.w), 0)
            self.compressed_bits.append((index, 0))
            self.max_index = max(self.max_index, index)
        return self._finalize_compression()

    def decode(self, compressed_data: bytes, output_size: int) -> List[int]:
        return self.decompress(compressed_data)
    
    
    def get_best_next_symbols(self):
        # Convert context to tuple for dictionary lookup
        context_tuple = tuple(self.w)
        matches = {}
        
        # Iterate over all dictionary entries
        for phrase in self.dictionary:
            # Check if the phrase extends the context
            if phrase[:len(context_tuple)] == context_tuple and len(phrase) > len(context_tuple):
                next_symbol = phrase[len(context_tuple)]
                match_length = len(phrase)
                if next_symbol < 7:
                    matches[next_symbol] = match_length
        
        # Sort the matches by longest match length
        sorted_matches = sorted(matches.items(), key=lambda x: -x[1])
        
        # Get the top 5 best next symbols
        best_symbols = [symbol for symbol, _ in sorted_matches[:16]]
        
        return best_symbols if best_symbols else [0]
    



if __name__ == '__main__':
    encoder = LZ78Encoder()
    input_data = [1,1, 2, 3, 1,2,3, 4,1,2,3,4, 1, 2, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    compressed_data = encoder.compress(input_data)
    c = encoder.get_best_next_symbol()
    print(c)