import numpy as np
import pickle

def load_pitch(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    return data
