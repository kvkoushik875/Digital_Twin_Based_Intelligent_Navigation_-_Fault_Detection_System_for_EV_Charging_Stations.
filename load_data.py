import pandas as pd
import os

def load_dataset(path):
    print("Loading dataset from:", path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    return pd.read_csv(path)
