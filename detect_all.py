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


def _load_artifacts():
    meta = json.loads((ARTIFACTS_DIR / "meta.json").read_text())
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    kmeans = joblib.load(ARTIFACTS_DIR / "fault_clusters.joblib")
    model = SensorAutoencoder(n_features=meta["n_features"])
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "autoencoder.pt", map_location="cpu"))
    model.eval()
    return model, scaler, kmeans, meta


def _severity(error, meta):
    """Boundaries are real percentiles (P95/P99/P99.9) of the model's own
    reconstruction-error distribution on validation data - computed in
    train.py, not arbitrary multipliers.
    """
    if error >= meta["failure_threshold"]:
        return "FAILURE"
    if error >= meta["critical_threshold"]:
        return "CRITICAL"
    return "WARNING"


AI_FAULT_TYPES = None  # populated per-run from meta["cluster_names"] in detect_all()


def detect_all():
    """Score every row in sensor_data with the trained autoencoder and
    replace the autoencoder-generated Fault records with the current
    run's results.

    fault_type comes from a KMeans model trained on the autoencoder's own
    per-feature error vectors (fault_engine/train.py) - which anomalous
    point belongs to which fault pattern is decided by unsupervised
    clustering, not a per-point rule. severity comes from real percentile
    boundaries of the model's error distribution, not fixed multipliers.

    Each run reflects what the *current* model considers faulty, not an
    ever-growing history: retraining shifts the threshold slightly, so
    without clearing stale flags first, every retrain would just add more
    rows on top of the last run's instead of superseding them. Faults
    logged by other sources (e.g. the rule-based fallback) are untouched,
    since only fault_type values this module can produce are cleared.
    """
    model, scaler, kmeans, meta = _load_artifacts()
    feature_columns = meta["feature_columns"]
    threshold = meta["threshold"]
    cluster_names = meta["cluster_names"]
    ai_fault_types = set(cluster_names)

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

    # Real per-feature P99 error across every real station, computed as
    # a side effect of the scoring pass above - used by
    # fault_engine/station_metrics.py to turn one station's per-feature
    # error into an honest 0-100 stability % for the frontend, without a
    # separate fabricated scale.
    feature_error_p99 = {
        col: float(np.percentile(per_feature_squared_error[:, idx], 99))
        for idx, col in enumerate(feature_columns)
    }
    (ARTIFACTS_DIR / "feature_error_p99.json").write_text(json.dumps(feature_error_p99))

    cluster_ids = (
        kmeans.predict(per_feature_squared_error[flagged_indices])
        if len(flagged_indices) > 0
        else np.array([], dtype=int)
    )

    db = SessionLocal()
    try:
        db.query(Fault).filter(Fault.fault_type.in_(ai_fault_types)).delete(
            synchronize_session=False
        )

        detected_at = datetime.now(timezone.utc)
        for pos, i in enumerate(flagged_indices):
            db.add(Fault(
                station_id=int(df.loc[i, "station_id"]),
                fault_type=cluster_names[cluster_ids[pos]],
                fault_score=float(errors[i]),
                severity=_severity(float(errors[i]), meta),
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
