"""
Per-station "stability" breakdown for the Health Assessment screen -
Voltage/Temperature/Power stability + Efficiency, each 0-100%.

Reuses the same trained fault-detection autoencoder as detect_all.py.
Each of the 16 real sensor features already produces its own
reconstruction error inside the model; this just reads that per-feature
error for one station and normalizes it against that same feature's
real P99 error (feature_error_p99.json, saved by detect_all() from the
same full-table scoring pass it already runs) - not a separately
fabricated scale.

METRIC_GROUPS is an interpretive grouping of which raw features map to
which headline label the frontend shows - a labeling choice, not a
different computation.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.database import SessionLocal
from app.models import SensorData

try:
    from .model import SensorAutoencoder
except ImportError:
    from model import SensorAutoencoder

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

METRIC_GROUPS = {
    "voltage_stability": ["voltage", "current"],
    "temperature_stability": ["battery_temp", "ambient_temp", "temperature"],
    "power_stability": ["charging_rate_kw", "energy_consumed_kwh", "battery_capacity_kwh"],
    "efficiency": ["efficiency", "degradation_rate", "charging_cycles"],
}


def _load_artifacts():
    meta = json.loads((ARTIFACTS_DIR / "meta.json").read_text())
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    model = SensorAutoencoder(n_features=meta["n_features"])
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "autoencoder.pt", map_location="cpu"))
    model.eval()

    p99_path = ARTIFACTS_DIR / "feature_error_p99.json"
    if not p99_path.exists():
        raise FileNotFoundError(
            "feature_error_p99.json missing - run fault_engine/detect_all.py once first"
        )
    percentiles = json.loads(p99_path.read_text())
    return model, scaler, meta, percentiles


def get_station_stability_metrics(station_id):
    model, scaler, meta, percentiles = _load_artifacts()
    feature_columns = meta["feature_columns"]

    db = SessionLocal()
    try:
        sensor = db.query(SensorData).filter(SensorData.station_id == station_id).first()
    finally:
        db.close()

    if sensor is None:
        return None

    values = [getattr(sensor, col) for col in feature_columns]
    if any(v is None for v in values):
        return None

    X = np.array([values], dtype=np.float32)
    X_scaled = scaler.transform(X).astype(np.float32)

    with torch.no_grad():
        t = torch.from_numpy(X_scaled)
        recon = model(t)
        per_feature_error = ((t - recon) ** 2).numpy()[0]

    error_by_feature = dict(zip(feature_columns, per_feature_error.tolist()))

    def stability_pct(feature):
        p99 = percentiles.get(feature) or 1e-9
        ratio = error_by_feature[feature] / p99
        return 100 * max(0.0, 1.0 - min(ratio, 1.0))

    result = {}
    for group_name, features in METRIC_GROUPS.items():
        present = [f for f in features if f in error_by_feature]
        result[group_name] = round(sum(stability_pct(f) for f in present) / len(present), 1)

    return result
