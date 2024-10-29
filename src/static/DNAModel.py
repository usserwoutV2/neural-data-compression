import numpy as np
import os
import sys
from typing import List
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train import create_and_train_model, load_model, save_model

class DNAModel:
    def __init__(self, input_string: str = None, model_path: str = 'char_model.keras', lookup_path: str = 'char_lookup.pkl'):
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', model_path))
        lookup_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', lookup_path))
        self.model_path = model_path
        self.lookup_path = lookup_path
        self.model = None
        self.char_to_index = None
        self.index_to_char = None
        print(model_path)
        if input_string and not os.path.exists(model_path):
            self._train_new_model(input_string)
        else:
            self._load_existing_model()

    def _train_new_model(self, input_string: str):
        print("Training new model...")
        self.model, self.char_to_index, self.index_to_char = create_and_train_model(input_string)
        save_model(self.model, self.char_to_index, self.index_to_char, self.model_path, self.lookup_path)
        print("Model trained and saved.")

    def _load_existing_model(self):
        print("Loading existing model...")
        self.model, self.char_to_index, self.index_to_char = load_model(self.model_path, self.lookup_path)
        print("Model loaded.")

    def predict_next_chars(self, input_string: str, sequence_length: int = 20) -> List[str]:
        input_indices = np.array([[self.char_to_index.get(char, 0) for char in input_string[-sequence_length:]]])
        predictions = self.model.predict(input_indices, verbose=0)[0]
<<<<<<< HEAD
        end_time = time.time()
        #print(f"Prediction took {end_time - start_time:.2f} seconds.")
=======
>>>>>>> a857832 (Updated the dynamic compressor)
        top_indices = predictions.argsort()[-4:][::-1]
        return [self.index_to_char[idx] for idx in top_indices]