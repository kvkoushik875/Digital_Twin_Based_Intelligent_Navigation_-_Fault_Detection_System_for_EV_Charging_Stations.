"""
Trains the SensorAutoencoder on ALL rows pulled live from the sensor_data
table (unsupervised - no normal/anomaly label available), picks a
fault-detection threshold from validation reconstruction error, and
sanity-checks detection with synthetic fault injection. Saves artifacts
to fault_engine/artifacts/.

Run from the project root:
    python -m fault_engine.train --epochs 200
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.database import engine

try:
    from .data_prep import load_and_prepare
    from .model import SensorAutoencoder
except ImportError:
    from data_prep import load_and_prepare
    from model import SensorAutoencoder

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reconstruction_errors(model, X, device, batch_size=1024):
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).to(device)
            recon = model(batch)
            err = torch.mean((batch - recon) ** 2, dim=1)
            errors.append(err.cpu().numpy())
    return np.concatenate(errors)


def train(args):
    device = get_device()
    print(f"Using device: {device}")

    df = pd.read_sql_table("sensor_data", engine)
    data = load_and_prepare(df)
    X_train, X_val = data["X_train"], data["X_val"]
    n_features = X_train.shape[1]
    print(f"n_features={n_features}  train={len(X_train)}  val={len(X_val)}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = SensorAutoencoder(n_features=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    X_val_t = torch.from_numpy(X_val).to(device)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), X_val_t).item()

        train_loss = float(np.mean(epoch_losses))
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    val_errors = reconstruction_errors(model, X_val, device)
    threshold = float(np.percentile(val_errors, args.threshold_percentile))
    print(f"\nThreshold (P{args.threshold_percentile} of validation error): {threshold:.6f}")

    print("\n=== Sanity check: synthetic fault injection ===")
    sample = X_val[:300].copy()
    injected = sample.copy()
    temp_idx = data["feature_columns"].index("battery_temp")
    current_idx = data["feature_columns"].index("current")
    injected[:, temp_idx] += 5.0
    injected[:, current_idx] += 5.0
    base_err = reconstruction_errors(model, sample, device)
    fault_err = reconstruction_errors(model, injected, device)
    detect_rate = float((fault_err > threshold).mean())
    print(f"Mean error on normal samples:         {base_err.mean():.4f}")
    print(f"Mean error on injected-fault samples: {fault_err.mean():.4f}")
    print(f"Detection rate on injected faults:    {detect_rate*100:.1f}%")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ARTIFACTS_DIR / "autoencoder.pt")
    joblib.dump(data["scaler"], ARTIFACTS_DIR / "scaler.joblib")

    meta = {
        "feature_columns": data["feature_columns"],
        "threshold": threshold,
        "threshold_percentile": args.threshold_percentile,
        "n_features": n_features,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "synthetic_fault_detection_rate": detect_rate,
    }
    (ARTIFACTS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nSaved model weights to: {ARTIFACTS_DIR / 'autoencoder.pt'}")
    print(f"Saved scaler to: {ARTIFACTS_DIR / 'scaler.joblib'}")
    print(f"Saved metadata/threshold to: {ARTIFACTS_DIR / 'meta.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--threshold_percentile", type=float, default=95.0)
    args = parser.parse_args()
    train(args)
