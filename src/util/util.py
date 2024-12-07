import numpy as np
import random
import torch
import math
from collections import Counter
from typing import List


def set_seed(seed: int):
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
    # Set the seed for NumPy
    np.random.seed(seed)
    
    # Set the seed for Python's built-in random module
    random.seed(seed)
    
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def load_dataset(name: str, char_count: int = None):
    
    if name == "mozilla":
        mode = "rb"
    else:
        mode = "r"
        name = name + ".txt"
    
    with open(f"datasets/data/{name}", mode) as f:
        data = f.read()
    if char_count is not None:
        data = data[:char_count]
    return data



def calculate_entropy(byte_data: str) -> float:

    
    # Calculate the frequency of each byte
    byte_counts = Counter(byte_data)
    total_bytes = len(byte_data)
    
    # Calculate the probability of each byte
    probabilities = [count / total_bytes for count in byte_counts.values()]
    
    # Calculate the entropy using the Shannon entropy formula
    entropy = -sum(p * math.log2(p) for p in probabilities)
    
    return entropy

def calculate_entropy_list(int_data: List[int]) -> float:
    # Calculate the frequency of each integer
    int_counts = Counter(int_data)
    total_ints = len(int_data)
    
    # Calculate the probability of each integer
    probabilities = [count / total_ints for count in int_counts.values()]
    
    # Calculate the entropy using the Shannon entropy formula
    entropy = -sum(p * math.log2(p) for p in probabilities)
    
    return entropy