import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.modeling.lstm_model import FaultLSTM


def train_pytorch_model(X, y, batch_size=32, epochs=10, lr=0.001):
    # -----------------------------
    # 1. Encode labels
    # -----------------------------
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # -----------------------------
    # 2. Convert to tensors
    # -----------------------------
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    # -----------------------------
    # 3. Create DataLoader
    # -----------------------------
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # -----------------------------
    # 4. Model definition
    # -----------------------------
    input_size = X.shape[2]
    hidden_size = 64
    num_classes = len(np.unique(y_encoded))

    model = FaultLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
        num_layers=2,
        dropout=0.3,
        bidirectional=False
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    # -----------------------------
    # 5. Training loop
    # -----------------------------
    for epoch in range(epochs):
        epoch_loss = 0.0

        for Xb, yb in loader:
            optimizer.zero_grad()

            outputs = model(Xb)

            # Shape validation (prevents backward crash)
            if outputs.shape[0] != yb.shape[0]:
                raise ValueError(
                    f"Shape mismatch: outputs={outputs.shape}, labels={yb.shape}"
                )

            loss = criterion(outputs, yb)
            loss.backward()

            # Gradient clipping (critical for LSTM stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    return model, le
