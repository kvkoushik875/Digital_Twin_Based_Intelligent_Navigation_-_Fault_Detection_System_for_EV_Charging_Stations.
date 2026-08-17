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
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.database import engine

try:
    from .data_prep import load_and_prepare
    from .features import FEATURE_FAULT_LABELS
    from .model import SensorAutoencoder
except ImportError:
    from data_prep import load_and_prepare
    from features import FEATURE_FAULT_LABELS
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


def per_feature_squared_errors(model, X, device, batch_size=1024):
    """Same as reconstruction_errors but keeps the per-feature breakdown
    (N, n_features) instead of averaging to one number per row - this is
    what the fault-type clustering groups on.
    """
    model.eval()
    all_errors = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).to(device)
            recon = model(batch)
            err = (batch - recon) ** 2
            all_errors.append(err.cpu().numpy())
    return np.concatenate(all_errors, axis=0)


def name_cluster(centroid, feature_columns, dominance_threshold=0.4):
    """Names a discovered fault cluster from its centroid's dominant
    feature - applied once per cluster (a handful of them), not per
    data point, so the clustering itself (which point belongs to which
    fault pattern) is fully unsupervised; only the human-readable label
    for each already-discovered group uses this rule.
    """
    total = centroid.sum()
    if total <= 0:
        return "MULTIVARIATE_SENSOR_ANOMALY"
    top_idx = int(np.argmax(centroid))
    top_share = centroid[top_idx] / total
    if top_share < dominance_threshold:
        return "MULTIVARIATE_SENSOR_ANOMALY"
    return FEATURE_FAULT_LABELS.get(feature_columns[top_idx], "MULTIVARIATE_SENSOR_ANOMALY")


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
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\nCompleted all {args.epochs} epochs. Best val_loss={best_val_loss:.5f}")
    model.load_state_dict(best_state)

    val_errors = reconstruction_errors(model, X_val, device)
    threshold = float(np.percentile(val_errors, args.threshold_percentile))
    critical_threshold = float(np.percentile(val_errors, 99.0))
    failure_threshold = float(np.percentile(val_errors, 99.9))
    print(f"\nThreshold (P{args.threshold_percentile} of validation error): {threshold:.6f}")
    print(f"Critical threshold (P99.0): {critical_threshold:.6f}")
    print(f"Failure threshold (P99.9): {failure_threshold:.6f}")

    print("\n=== Discovering fault-type clusters (unsupervised) ===")
    feature_columns = data["feature_columns"]
    X_full = data["scaler"].transform(
        df.dropna(subset=feature_columns).reset_index(drop=True)[feature_columns].values
    ).astype(np.float32)
    full_per_feature_err = per_feature_squared_errors(model, X_full, device)
    full_errors = full_per_feature_err.mean(axis=1)
    flagged_mask = full_errors > threshold
    flagged_matrix = full_per_feature_err[flagged_mask]

    n_clusters = min(args.n_fault_clusters, max(1, len(flagged_matrix)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(flagged_matrix)
    cluster_names = [
        name_cluster(kmeans.cluster_centers_[c], feature_columns)
        for c in range(n_clusters)
    ]
    for c, name in enumerate(cluster_names):
        print(f"  Cluster {c}: {name}")

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
    joblib.dump(kmeans, ARTIFACTS_DIR / "fault_clusters.joblib")

    meta = {
        "feature_columns": data["feature_columns"],
        "threshold": threshold,
        "critical_threshold": critical_threshold,
        "failure_threshold": failure_threshold,
        "threshold_percentile": args.threshold_percentile,
        "n_features": n_features,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "synthetic_fault_detection_rate": detect_rate,
        "n_fault_clusters": n_clusters,
        "cluster_names": cluster_names,
    }
    (ARTIFACTS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nSaved model weights to: {ARTIFACTS_DIR / 'autoencoder.pt'}")
    print(f"Saved scaler to: {ARTIFACTS_DIR / 'scaler.joblib'}")
    print(f"Saved fault clusters to: {ARTIFACTS_DIR / 'fault_clusters.joblib'}")
    print(f"Saved metadata/threshold to: {ARTIFACTS_DIR / 'meta.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold_percentile", type=float, default=95.0)
    parser.add_argument("--n_fault_clusters", type=int, default=6)
    args = parser.parse_args()
    train(args)
