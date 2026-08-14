import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.database import SessionLocal, engine
from app.models import Fault

try:
    from .model import SensorAutoencoder
except ImportError:
    from model import SensorAutoencoder

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

# Maps to 3 of the 5 fault categories named in the project abstract
# (voltage instability, excessive power loss, charger overheating).
# Communication failures and interrupted charging sessions have no
# corresponding column in sensor_data, so they aren't detectable here yet.
FEATURE_FAULT_LABELS = {
    "soc_percent": "SOC_ANOMALY",
    "voltage": "VOLTAGE_INSTABILITY",
    "current": "CURRENT_ANOMALY",
    "battery_temp": "CHARGER_OVERHEATING",
    "ambient_temp": "CHARGER_OVERHEATING",
    "charging_duration_min": "CHARGING_DURATION_ANOMALY",
    "degradation_rate": "DEGRADATION_ANOMALY",
    "efficiency": "EXCESSIVE_POWER_LOSS",
    "charging_cycles": "CHARGING_CYCLES_ANOMALY",
    "battery_capacity_kwh": "BATTERY_CAPACITY_ANOMALY",
    "energy_consumed_kwh": "EXCESSIVE_POWER_LOSS",
    "charging_duration_hours": "CHARGING_DURATION_ANOMALY",
    "charging_rate_kw": "EXCESSIVE_POWER_LOSS",
    "soc_start": "SOC_ANOMALY",
    "soc_end": "SOC_ANOMALY",
    "temperature": "CHARGER_OVERHEATING",
}


def _load_artifacts():
    meta = json.loads((ARTIFACTS_DIR / "meta.json").read_text())
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    model = SensorAutoencoder(n_features=meta["n_features"])
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "autoencoder.pt", map_location="cpu"))
    model.eval()
    return model, scaler, meta


def _severity(error_to_threshold_ratio):
    if error_to_threshold_ratio >= 2.5:
        return "FAILURE"
    if error_to_threshold_ratio >= 1.5:
        return "CRITICAL"
    return "WARNING"


def _fault_type(per_feature_squared_error, feature_columns):
    total = per_feature_squared_error.sum()
    if total <= 0:
        return "MULTIVARIATE_SENSOR_ANOMALY"
    top_idx = int(np.argmax(per_feature_squared_error))
    top_share = per_feature_squared_error[top_idx] / total
    if top_share < 0.4:
        return "MULTIVARIATE_SENSOR_ANOMALY"
    return FEATURE_FAULT_LABELS.get(feature_columns[top_idx], "MULTIVARIATE_SENSOR_ANOMALY")


AI_FAULT_TYPES = set(FEATURE_FAULT_LABELS.values()) | {"MULTIVARIATE_SENSOR_ANOMALY"}


def detect_all():
    """Score every row in sensor_data with the trained autoencoder and
    replace the autoencoder-generated Fault records with the current
    run's results.

    Each run reflects what the *current* model considers faulty, not an
    ever-growing history: retraining shifts the threshold slightly, so
    without clearing stale flags first, every retrain would just add more
    rows on top of the last run's instead of superseding them. Faults
    logged by other sources (e.g. the rule-based fallback) are untouched,
    since only fault_type values this module can produce are cleared.
    """
    model, scaler, meta = _load_artifacts()
    feature_columns = meta["feature_columns"]
    threshold = meta["threshold"]

    df = pd.read_sql_table("sensor_data", engine)
    df = df.dropna(subset=feature_columns).reset_index(drop=True)
    if df.empty:
        return {"scored": 0, "flagged": 0, "created": 0}

    X = df[feature_columns].values.astype(np.float32)
    X_scaled = scaler.transform(X).astype(np.float32)

    with torch.no_grad():
        t = torch.from_numpy(X_scaled)
        recon = model(t)
        per_feature_squared_error = (t - recon).numpy() ** 2

    errors = per_feature_squared_error.mean(axis=1)
    flagged_indices = np.where(errors > threshold)[0]

    db = SessionLocal()
    try:
        db.query(Fault).filter(Fault.fault_type.in_(AI_FAULT_TYPES)).delete(
            synchronize_session=False
        )

        detected_at = datetime.now(timezone.utc)
        for i in flagged_indices:
            db.add(Fault(
                station_id=int(df.loc[i, "station_id"]),
                fault_type=_fault_type(per_feature_squared_error[i], feature_columns),
                fault_score=float(errors[i]),
                severity=_severity(float(errors[i] / threshold)),
                detected_at=detected_at,
            ))

        db.commit()
    finally:
        db.close()

    return {
        "scored": int(len(df)),
        "flagged": int(len(flagged_indices)),
        "created": int(len(flagged_indices)),
        "threshold": threshold,
    }
