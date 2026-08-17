"""
Trains a small autoencoder - a genuine deep learning model, same
unsupervised principle as fault_engine's SensorAutoencoder - on each
station's health engine output, then immediately scores every station
with it. No separate training step or persisted artifacts to run first:
every call to assess_all() trains fresh from whatever is currently in
health_assessment and writes the results.

Trains only on health_assessment (station_status, critical_faults,
warning_faults) - not on raw sensor_data - so predictive maintenance
depends on the health engine's own verdict, not a second, independent
read of the sensors. Needs no failure-date labels (which this project
doesn't have): purely unsupervised, like the fault engine.

KNOWN LIMITATION (accepted, intentional - not a bug): reconstruction
error measures "how hard is this point to reconstruct," not "how
severe is this station's status." A CRITICAL health status can be
triggered by a single critical fault, while MAINTENANCE_REQUIRED can
involve several warning faults - so MAINTENANCE_REQUIRED stations
often have larger raw fault-count values than CRITICAL ones, even
though CRITICAL is the worse outcome. The autoencoder has no concept
of which status is "worse," only which points are numerically further
from what it learned as typical - so maintenance_priority tiers do
NOT reliably rank CRITICAL above MAINTENANCE_REQUIRED above MONITOR.
This was a deliberate choice to keep the model purely unsupervised
rather than add a rule-based ordering floor.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.database import SessionLocal, engine
from app.models import ChargingStation, PredictiveMaintenance

try:
    from .model import MaintenanceAutoencoder
except ImportError:
    from model import MaintenanceAutoencoder

FEATURE_COLUMNS = ["status_level", "critical_faults", "warning_faults"]

# Ordinal encoding of the health engine's own status ladder, so it can be
# used as a numeric feature alongside its fault counts.
STATUS_LEVEL = {
    "HEALTHY": 0,
    "MONITOR": 1,
    "MAINTENANCE_REQUIRED": 2,
    "CRITICAL": 3,
}

# Percentile cutoffs on the model's own reconstruction-error
# distribution - real statistics from the trained model, not arbitrary
# numbers. Mirrors the same percentile-threshold approach fault_engine
# uses for its severity tiers.
STATUS_PERCENTILES = {
    "IMMEDIATE_MAINTENANCE": 99.0,
    "SCHEDULE_MAINTENANCE": 95.0,
    "MONITOR": 80.0,
}

STATUS_DAYS_TO_FAILURE = {
    "IMMEDIATE_MAINTENANCE": 7,
    "SCHEDULE_MAINTENANCE": 30,
    "MONITOR": 90,
    "HEALTHY": 365,
}

STATUS_RECOMMENDATION = {
    "IMMEDIATE_MAINTENANCE": "Immediate inspection required - autoencoder ranks this station in the top 1% least like a normal health profile.",
    "SCHEDULE_MAINTENANCE": "Schedule maintenance soon - autoencoder ranks this station in the top 5% least like a normal health profile.",
    "MONITOR": "Monitor closely - autoencoder ranks this station in the top 20% least like a normal health profile.",
    "HEALTHY": "No immediate action needed.",
}


def load_training_frame():
    query = """
        SELECT DISTINCT ON (station_id)
            station_id, station_status, critical_faults, warning_faults
        FROM health_assessment
        ORDER BY station_id, assessment_time DESC
    """
    df = pd.read_sql(query, engine)
    df["status_level"] = df["station_status"].map(STATUS_LEVEL)
    return df


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


def _train_model(df, epochs=100, batch_size=256, lr=1e-3):
    device = get_device()
    # Fixed seed: training happens fresh on every assess_all() call (no
    # persisted artifacts), and this model is small enough that random
    # initialization can occasionally scramble the reconstruction-error
    # ordering between low-sample minority classes (e.g. CRITICAL vs
    # MAINTENANCE_REQUIRED) - a fixed seed keeps results stable and
    # reproducible across runs instead of silently flipping.
    torch.manual_seed(42)

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    model = MaintenanceAutoencoder(n_features=X_scaled.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_scaled)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

    errors = reconstruction_errors(model, X_scaled, device)
    thresholds = {
        status: float(np.percentile(errors, p))
        for status, p in STATUS_PERCENTILES.items()
    }
    return model, scaler, thresholds, errors, device


def _status_for(error, thresholds):
    # Strict '>' matters here: a large majority of stations share the
    # exact same health profile (HEALTHY, 0 faults), so they share the
    # exact same reconstruction error - which can land exactly on a
    # percentile threshold. '>=' would incorrectly sweep that whole
    # majority into the tier above; '>' correctly keeps them below it.
    if error > thresholds["IMMEDIATE_MAINTENANCE"]:
        return "IMMEDIATE_MAINTENANCE"
    if error > thresholds["SCHEDULE_MAINTENANCE"]:
        return "SCHEDULE_MAINTENANCE"
    if error > thresholds["MONITOR"]:
        return "MONITOR"
    return "HEALTHY"


def assess_all():
    """Trains fresh on the current health_assessment data and scores
    every station in one call. Replaces the previous run's results
    rather than accumulating - current-priority snapshot, not a trend
    log, same reasoning as fault_engine/health_engine.
    """
    df = load_training_frame()
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    if df.empty:
        return {"assessed": 0, "immediate_maintenance": 0, "schedule_maintenance": 0, "monitor": 0, "healthy": 0}

    model, scaler, thresholds, errors, device = _train_model(df)

    db = SessionLocal()
    try:
        known_station_ids = {row[0] for row in db.query(ChargingStation.station_id).all()}

        db.query(PredictiveMaintenance).delete()

        now = datetime.now(timezone.utc)
        summary = {"assessed": 0, "immediate_maintenance": 0, "schedule_maintenance": 0, "monitor": 0, "healthy": 0}
        records = []

        for i in range(len(df)):
            station_id = int(df.loc[i, "station_id"])
            if station_id not in known_station_ids:
                continue

            status = _status_for(float(errors[i]), thresholds)

            records.append(PredictiveMaintenance(
                station_id=station_id,
                predicted_failure_days=STATUS_DAYS_TO_FAILURE[status],
                maintenance_priority=status,
                recommendation=STATUS_RECOMMENDATION[status],
                created_at=now,
            ))
            summary["assessed"] += 1
            summary[status.lower()] += 1

        db.add_all(records)
        db.commit()
    finally:
        db.close()

    return summary


if __name__ == "__main__":
    print(assess_all())
