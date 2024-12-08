from util import load_dataset
import time
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from encoders.Encoder import Encoder


def main():
  for data_set in ["webster", "bible", "text8", "mozilla", "chr20", "celegchr"]:
    input_string = load_dataset(data_set, 1_000_000)

    for encode_method in [
      "arithmetic",
      "huffman",
      "lzma",
      "brotli",
      "gzip",
      "bz2",
      "zlib",
    ]:
      start = time.perf_counter()
      encoder = Encoder(method=encode_method)
      compressed = encoder._encode_str(input_string)
      end = time.perf_counter()

      print(
        f"[{data_set}] Size: ({encode_method}): {len(compressed)} bytes. Time: {end-start:.6f} seconds"
      )


if __name__ == "__main__":
  main()
