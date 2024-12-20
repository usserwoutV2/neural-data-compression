import numpy as np
import os
import sys
from typing import List
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train import create_and_train_model, load_model, save_model
from StaticModel import StaticModel

LEGAL_CHARACTERS = [chr(i) for i in range(32, 127)] + ['\n']  # ASCII characters from space to ~

class EnglishTextModel(StaticModel):
    # paths need to be edited when not running on the VSC
    def __init__(self, input_string: str = None, model_path: str = os.path.join(os.environ['VSC_DATA'], 'char_model.keras'), lookup_path: str = os.path.join(os.environ['VSC_DATA'], 'char_lookup.pkl')):
        super().__init__(LEGAL_CHARACTERS, input_string, model_path, lookup_path)