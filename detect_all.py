"""Detect faults across all PostgreSQL sensor records."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sqlalchemy import text

from ai_engine.fault_engine import FaultDetectionEngine
from database import engine


# =========================================================
# Configuration
# =========================================================

BATCH_SIZE = 2048

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"
RESULTS_PATH = MODEL_DIR / "anomaly_results.csv"


# =========================================================
# Load database records
# =========================================================

def load_all_records(features):
    """Load every record from sensor_data."""

    feature_columns = ", ".join(features)

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

    print("Loading all sensor records...")

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
            "Invalid or infinite sensor values detected."
        )

    print(f"Loaded records: {len(dataframe)}")

    return dataframe


# =========================================================
# Calculate anomaly scores
# =========================================================

def calculate_scores(dataframe, fault_engine):
    """Calculate Autoencoder reconstruction errors in batches."""

    predictor = fault_engine.predictor
    features = predictor.features

    scaled_values = predictor.scaler.transform(
        dataframe[features]
    )

    all_scores = []
    total_records = len(dataframe)

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
            device=predictor.device,
        )

        with torch.inference_mode():
            reconstructed = predictor.model(batch)

            batch_scores = torch.mean(
                torch.square(
                    batch - reconstructed
                ),
                dim=1,
            )

        all_scores.extend(
            batch_scores
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        print(
            f"Processed {end}/{total_records}",
            end="\r",
            flush=True,
        )

    print()

    return np.asarray(
        all_scores,
        dtype=np.float64,
    )


# =========================================================
# Severity classification
# =========================================================

def classify_scores(scores, fault_engine):
    """Classify every anomaly score."""

    predictor = fault_engine.predictor

    return [
        predictor.classify_fault(float(score))
        for score in scores
    ]


# =========================================================
# Fault-type diagnosis
# =========================================================

def identify_fault_types_for_all(
    dataframe,
    fault_engine,
):
    """Identify probable fault types for anomalous records."""

    fault_type_results = []

    for row in dataframe.itertuples(index=False):
        row_data = row._asdict()

        if not bool(row_data["is_anomaly"]):
            fault_type_results.append("NO_FAULT")
            continue

        identified_faults = (
            fault_engine.identify_fault_types(
                row_data
            )
        )

        if identified_faults:
            fault_type_results.append(
                ", ".join(identified_faults)
            )
        else:
            fault_type_results.append(
                "MULTIVARIATE_SENSOR_ANOMALY"
            )

    return fault_type_results


# =========================================================
# Save results to CSV
# =========================================================

def save_results_csv(dataframe):
    """Save complete detection results."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_columns = [
        "sensor_record_number",
        "station_id",
        "anomaly_score",
        "fault_type",
        "fault_severity",
        "is_anomaly",
    ]

    dataframe[output_columns].to_csv(
        RESULTS_PATH,
        index=False,
    )

    print(f"CSV saved: {RESULTS_PATH}")


# =========================================================
# Insert anomalies into faults table
# =========================================================

def save_anomalies_to_database(anomalies):
    """Insert anomalous records into the faults table."""

    if anomalies.empty:
        print("No anomalies found.")
        return 0

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

    for row in anomalies.itertuples(index=False):
        records.append(
            {
                "station_id": int(row.station_id),
                "fault_type": str(row.fault_type),
                "fault_score": float(
                    row.anomaly_score
                ),
                "severity": str(
                    row.fault_severity
                ),
            }
        )

    print(f"Inserting {len(records)} anomalies...")

    with engine.begin() as connection:
        connection.execute(
            insert_query,
            records,
        )

    print(
        f"Inserted {len(records)} anomalies "
        "into the faults table."
    )

    return len(records)


# =========================================================
# Main detection process
# =========================================================

def detect_all():
    """Run complete fault detection on every sensor record."""

    print("Starting complete fault detection...")

    fault_engine = FaultDetectionEngine()
    predictor = fault_engine.predictor

    print(f"Inference device: {predictor.device}")

    dataframe = load_all_records(
        predictor.features
    )

    anomaly_scores = calculate_scores(
        dataframe,
        fault_engine,
    )

    dataframe["anomaly_score"] = anomaly_scores

    dataframe["fault_severity"] = (
        classify_scores(
            anomaly_scores,
            fault_engine,
        )
    )

    dataframe["is_anomaly"] = (
        dataframe["fault_severity"] != "NORMAL"
    )

    dataframe["fault_type"] = (
        identify_fault_types_for_all(
            dataframe,
            fault_engine,
        )
    )

    anomalies = dataframe.loc[
        dataframe["is_anomaly"]
    ].copy()

    normal_count = int(
        (
            dataframe["fault_severity"]
            == "NORMAL"
        ).sum()
    )

    warning_count = int(
        (
            dataframe["fault_severity"]
            == "WARNING"
        ).sum()
    )

    critical_count = int(
        (
            dataframe["fault_severity"]
            == "CRITICAL"
        ).sum()
    )

    failure_count = int(
        (
            dataframe["fault_severity"]
            == "FAILURE"
        ).sum()
    )

    print("\nDetection summary")
    print(f"NORMAL:   {normal_count}")
    print(f"WARNING:  {warning_count}")
    print(f"CRITICAL: {critical_count}")
    print(f"FAILURE:  {failure_count}")
    print(f"\nTotal records: {len(dataframe)}")
    print(f"Total anomalies: {len(anomalies)}")

    save_results_csv(dataframe)

    inserted_records = save_anomalies_to_database(
        anomalies
    )

    result = {
        "total_records": int(len(dataframe)),
        "normal": normal_count,
        "warning": warning_count,
        "critical": critical_count,
        "failure": failure_count,
        "total_anomalies": int(len(anomalies)),
        "inserted_records": int(inserted_records),
        "results_file": str(RESULTS_PATH),
    }

    print("\nComplete fault detection finished.")

    return result


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    summary = detect_all()

    print("\nFinal result:")
    print(summary)