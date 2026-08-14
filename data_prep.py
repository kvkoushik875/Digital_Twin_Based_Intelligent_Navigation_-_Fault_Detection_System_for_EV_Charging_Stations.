import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from .features import FEATURE_COLUMNS
except ImportError:
    from features import FEATURE_COLUMNS


def load_and_prepare(df, val_size=0.15, random_state=42):
    """Train/val split of ALL sensor_data rows - unsupervised, no
    normal/anomaly label to filter by. This relies on faults being a
    small minority of readings, so the autoencoder still learns the
    dominant "normal" pattern even with some anomalies mixed into
    training; this is standard practice for autoencoder anomaly
    detection when clean labels aren't available.
    """
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    train_df, val_df = train_test_split(df, test_size=val_size, random_state=random_state)

    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLUMNS].values)

    def scale(frame):
        return scaler.transform(frame[FEATURE_COLUMNS].values).astype(np.float32)

    return {
        "scaler": scaler,
        "X_train": scale(train_df),
        "X_val": scale(val_df),
        "feature_columns": FEATURE_COLUMNS,
        "n_total": len(df),
    }


if __name__ == "__main__":
    print(
        "data_prep.py is a support module, not an entry point.\n"
        "To train the model, run from the project root:\n"
        "    python -m fault_engine.train --epochs 200\n"
        "To run bulk fault detection:\n"
        "    python -c \"from fault_engine.detect_all import detect_all; print(detect_all())\""
    )
