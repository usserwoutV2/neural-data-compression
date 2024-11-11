import numpy as np
import random
import torch

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