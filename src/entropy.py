import math
from collections import Counter
from exampleData import sample4,sample2, sample1
from typing import List

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

# Example usage
if __name__ == "__main__":
    file_path = './datasets/files_to_be_compressed/chr20.txt'
    with open(file_path, 'rb') as file:
        byte_data = file.read()
        
    entropy = calculate_entropy(byte_data)
    print(f"Entropy of the file: {entropy:.4f} bits")