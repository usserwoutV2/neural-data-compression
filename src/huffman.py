import heapq
from collections import defaultdict
import bitarray
from itertools import count
from collections import Counter

class HuffmanCoding:
    def __init__(self):
        self.codes = {}
        self.reverse_mapping = {}
        self.count = count()

    def build_priority_queue(self, frequencies):
        heap = [[freq, next(self.count), symbol] for symbol, freq in enumerate(frequencies) if freq > 0]
        heapq.heapify(heap)
        return heap

    def build_tree(self, heap):
        while len(heap) > 1:
            freq1, _, left = heapq.heappop(heap)
            freq2, _, right = heapq.heappop(heap)
            heapq.heappush(heap, [freq1 + freq2, next(self.count), (left, right)])
        return heap[0][2] if heap else None

    def generate_codes(self, tree, prefix=0, depth=0):
        if isinstance(tree, int):  # Leaf node
            self.codes[tree] = (prefix, depth)
            self.reverse_mapping[(prefix, depth)] = tree
        else:
            left, right = tree
            self.generate_codes(left, prefix << 1, depth + 1)
            self.generate_codes(right, (prefix << 1) | 1, depth + 1)

    def encode_data(self, data):
        bit_array = bitarray.bitarray()
        for number in data:
            code, length = self.codes[number]
            bit_array.extend(f"{code:0{length}b}")
        return bit_array

    def decode_data(self, bit_array):
        current_prefix = 0
        current_depth = 0
        decoded = []
        for bit in bit_array:
            current_prefix = (current_prefix << 1) | bit
            current_depth += 1
            if (current_prefix, current_depth) in self.reverse_mapping:
                decoded.append(self.reverse_mapping[(current_prefix, current_depth)])
                current_prefix = 0
                current_depth = 0
        return decoded

    def compress(self, data, frequencies):
        # Build Huffman tree
        heap = self.build_priority_queue(frequencies)
        tree = self.build_tree(heap)
        self.generate_codes(tree)

        # Encode data
        encoded_bit_array = self.encode_data(data)
        return encoded_bit_array.tobytes()

    def decompress(self, compressed_data, frequencies):
        # Rebuild the Huffman tree
        heap = self.build_priority_queue(frequencies)
        tree = self.build_tree(heap)
        self.generate_codes(tree)

        # Decode the data
        bit_array = bitarray.bitarray()
        bit_array.frombytes(compressed_data)
        return self.decode_data(bit_array)


# ---------------------- #
class HuffmanCoding2:
    def __init__(self):
        self.codes = {}
        self.reverse_mapping = {}

    def build_frequency_table(self, text):
        return Counter(text)

    def build_priority_queue(self, freq_table):
        heap = [[weight, [symbol, ""]] for symbol, weight in freq_table.items()]
        heapq.heapify(heap)
        return heap

    def merge_nodes(self, heap):
        while len(heap) > 1:
            low1 = heapq.heappop(heap)
            low2 = heapq.heappop(heap)
            for pair in low1[1:]:
                pair[1] = "0" + pair[1]
            for pair in low2[1:]:
                pair[1] = "1" + pair[1]
            heapq.heappush(heap, [low1[0] + low2[0]] + low1[1:] + low2[1:])
        return heap

    def build_codes(self, heap):
        tree = heapq.heappop(heap)
        for symbol, code in tree[1:]:
            self.codes[symbol] = code
            self.reverse_mapping[code] = symbol

    def encode_text(self, text):
        encoded_text = ''.join(self.codes[char] for char in text)
        return bitarray.bitarray(encoded_text)

    def compress(self, text):
        # Build the Huffman tree and generate codes
        freq_table = self.build_frequency_table(text)
        heap = self.build_priority_queue(freq_table)
        heap = self.merge_nodes(heap)
        self.build_codes(heap)

        # Encode the text and convert to byte array
        encoded_text = self.encode_text(text)
        return encoded_text.tobytes()

    def decode_text(self, encoded_text):
        current_code = ""
        decoded_text = []

        for bit in encoded_text:
            current_code += bit
            if current_code in self.reverse_mapping:
                character = self.reverse_mapping[current_code]
                decoded_text.append(character)
                current_code = ""

        return ''.join(decoded_text)

    def decompress(self, compressed_data, freq_table):
        # Rebuild the Huffman tree and generate codes
        heap = self.build_priority_queue(freq_table)
        heap = self.merge_nodes(heap)
        self.build_codes(heap)

        # Decode the byte array to bit array
        encoded_text = bitarray.bitarray()
        encoded_text.frombytes(compressed_data)

        # Decode the text
        return self.decode_text(encoded_text)




    

if __name__ == "__main__":
    # Example Usage
    data = [0, 1, 1, 2, 3, 3, 3, 3, 0, 2, 3, 1]
    frequencies = [2, 3, 2, 5]  # Frequencies for numbers 0, 1, 2, 3

    huffman = HuffmanCoding()
    compressed = huffman.compress(data, frequencies)
    print(f"Compressed Byte Array: {compressed}")

    decompressed = huffman.decompress(compressed, frequencies)
    print(f"Decompressed Data: {decompressed}")
