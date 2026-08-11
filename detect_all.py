"""Detect anomalies in all sensor_data records."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sqlalchemy import text

from ai_engine.autoencoder import EVAutoencoder
from database import engine


# =========================================================
# Configuration and paths
# =========================================================

BATCH_SIZE = 2048

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"

MODEL_PATH = MODEL_DIR / "autoencoder.pt"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
METADATA_PATH = MODEL_DIR / "metadata.pkl"
RESULTS_PATH = MODEL_DIR / "anomaly_results.csv"


# =========================================================
# Load trained files
# =========================================================

def load_artifacts():
    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        METADATA_PATH,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise RuntimeError(
            "Missing trained-model files: "
            + ", ".join(missing_files)
        )

    metadata = joblib.load(METADATA_PATH)
    scaler = joblib.load(SCALER_PATH)

    return metadata, scaler


# =========================================================
# Load every sensor record
# =========================================================

def load_all_records(
    features: list[str],
) -> pd.DataFrame:

    feature_columns = ", ".join(features)

    # sensor_data has no id column, so ROW_NUMBER creates
    # a temporary record number for the CSV output.
    query = text(
        f"""
        SELECT
            ROW_NUMBER() OVER (
                ORDER BY station_id
            ) AS sensor_record_number,
            station_id,
            {feature_columns}
        FROM sensor_data
        ORDER BY station_id
        """
    )

    print("Loading sensor records...")

    with engine.connect() as connection:
        dataframe = pd.read_sql_query(
            sql=query,
            con=connection,
        )

    if dataframe.empty:
        raise RuntimeError(
            "The sensor_data table contains no records."
        )

    missing_columns = [
        feature
        for feature in features
        if feature not in dataframe.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f"Missing sensor columns: {missing_columns}"
        )

    if dataframe[features].isnull().any().any():
        raise RuntimeError(
            "NULL values exist in sensor features."
        )

    sensor_values = dataframe[
        features
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(sensor_values).all():
        raise RuntimeError(
            "Infinite or invalid sensor values detected."
        )

    print(f"Loaded records: {len(dataframe)}")

    return dataframe


# =========================================================
# Fault classification
# =========================================================

def classify_score(
    anomaly_score: float,
    metadata: dict,
) -> str:

    if anomaly_score >= metadata["failure_threshold"]:
        return "FAILURE"

    if anomaly_score >= metadata["critical_threshold"]:
        return "CRITICAL"

    if anomaly_score >= metadata["warning_threshold"]:
        return "WARNING"

    return "NORMAL"


# =========================================================
# Calculate anomaly scores
# =========================================================

def calculate_scores(
    dataframe: pd.DataFrame,
    features: list[str],
    scaler,
    model: EVAutoencoder,
    device: torch.device,
) -> np.ndarray:

    scaled_values = scaler.transform(
        dataframe[features]
    )

    all_scores = []
    total_records = len(scaled_values)

    for start in range(
        0,
        total_records,
        BATCH_SIZE,
    ):
        end = min(
            start + BATCH_SIZE,
            total_records,
        )

        batch = torch.tensor(
            scaled_values[start:end],
            dtype=torch.float32,
            device=device,
        )

        with torch.inference_mode():
            reconstructed = model(batch)

            batch_scores = torch.mean(
                torch.square(
                    batch - reconstructed
                ),
                dim=1,
            )

        all_scores.extend(
            batch_scores.cpu().numpy()
        )

        print(
            f"Processed {end}/{total_records}",
            end="\r",
            flush=True,
        )

    print()

    return np.asarray(all_scores)


# =========================================================
# Save complete results to CSV
# =========================================================

def save_csv_results(
    dataframe: pd.DataFrame,
) -> None:

    output_columns = [
        "sensor_record_number",
        "station_id",
        "anomaly_score",
        "fault_severity",
        "is_anomaly",
    ]

    dataframe[output_columns].to_csv(
        RESULTS_PATH,
        index=False,
    )

    print(f"CSV results saved: {RESULTS_PATH}")


# =========================================================
# Insert anomalies into faults table
# =========================================================

def save_anomalies_to_database(
    anomalies: pd.DataFrame,
) -> None:

    if anomalies.empty:
        print("No anomalies found.")
        return

    insert_query = text(
        """
        INSERT INTO faults (
            station_id,
            fault_type,
            fault_score,
            severity,
            detected_at
        )
        VALUES (
            :station_id,
            :fault_type,
            :fault_score,
            :severity,
            CURRENT_TIMESTAMP
        )
        """
    )

    records = []

    for row in anomalies.itertuples():
        records.append(
            {
                "station_id": int(row.station_id),
                "fault_type": "AUTOENCODER_ANOMALY",
                "fault_score": float(
                    row.anomaly_score
                ),
                "severity": str(
                    row.fault_severity
                ),
            }
        )

    print(
        f"Inserting {len(records)} anomalies..."
    )

    with engine.begin() as connection:
        connection.execute(
            insert_query,
            records,
        )

    print(
        f"Inserted {len(records)} anomalies "
        "into the faults table."
    )


# =========================================================
# Main process
# =========================================================

def detect_all() -> None:

    print("Starting bulk anomaly detection...")

    metadata, scaler = load_artifacts()

    features = metadata["features"]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Inference device: {device}")

    model = EVAutoencoder(
        input_dim=int(
            metadata["input_dim"]
        ),
        latent_dim=int(
            metadata["latent_dim"]
        ),
    ).to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()

    dataframe = load_all_records(
        features
    )

    anomaly_scores = calculate_scores(
        dataframe=dataframe,
        features=features,
        scaler=scaler,
        model=model,
        device=device,
    )

    dataframe["anomaly_score"] = (
        anomaly_scores
    )

    dataframe["fault_severity"] = [
        classify_score(
            float(score),
            metadata,
        )
        for score in anomaly_scores
    ]

    dataframe["is_anomaly"] = (
        dataframe["fault_severity"]
        != "NORMAL"
    )

    anomalies = dataframe[
        dataframe["is_anomaly"]
    ].copy()

    summary = (
        dataframe["fault_severity"]
        .value_counts()
        .reindex(
            [
                "NORMAL",
                "WARNING",
                "CRITICAL",
                "FAILURE",
            ],
            fill_value=0,
        )
    )

    print("\nDetection summary")
    print(summary.to_string())

    print(
        f"\nTotal records: {len(dataframe)}"
    )

    print(
        f"Total anomalies: {len(anomalies)}"
    )

    save_csv_results(dataframe)

    save_anomalies_to_database(
        anomalies
    )

    print(
        "\nBulk anomaly detection completed."
    )


if __name__ == "__main__":
    detect_all()