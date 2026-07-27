import numpy as np

def create_windows(data, labels, window=60):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(labels[i+window])
    return np.array(X), np.array(y)
