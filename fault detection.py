"""
Fault detection for EV charging stations using the full sensor dataset.

The dataset contains station metadata, live electrical readings, charging
session metrics, connector temperature, grid load, and a StatusType field.
This script combines:

1. Supervised ML fault classification from StatusType.
2. Rule-based sensor checks that explain why a station looks risky.
3. A full output CSV with known status, ML risk, sensor risk, and reasons.

Run:
    python ev_charging_fault_detection.py

Run with a different dataset:
    python ev_charging_fault_detection.py "path/to/ev_charging_full_sensor_dataset.csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_DATASET = (
    r"D:\Digital Twin based EV charging station navigations"
    r"\ev_charging_full_sensor_dataset.csv"
)
DEFAULT_RESULTS_FILE = "fault_detection_results.csv"
DEFAULT_MODEL_FILE = "ev_fault_detection_model.joblib"

NORMAL_STATUSES = {
    "Operational",
    "Currently Available (Automated Status)",
}

FAULT_STATUSES = {
    "Not Operational",
    "Temporarily Unavailable",
    "Partly Operational (Mixed)",
}

PLANNED_STATUS = "Planned For Future Date"

NUMERIC_FEATURES = [
    "MaxPowerKW",
    "FastChargeCount",
    "Voltage_V",
    "Current_A",
    "Session_Energy_kWh",
    "Temperature_C",
    "SoC_pct",
    "Session_Duration_min",
    "Charging_Efficiency_pct",
    "Connector_Temp_C",
    "Grid_Load_kW",
    "Computed_Power_kW",
    "Power_Mismatch_pct",
    "Connector_Temp_Rise_C",
    "Energy_Rate_kWh_per_min",
    "Power_Utilization_pct",
]

CATEGORICAL_FEATURES = [
    "Operator",
    "UsageType",
    "ConnectionTypes",
]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {
        "StationID",
        "Operator",
        "UsageType",
        "MaxPowerKW",
        "FastChargeCount",
        "ConnectionTypes",
        "StatusType",
        "Voltage_V",
        "Current_A",
        "Session_Energy_kWh",
        "Temperature_C",
        "SoC_pct",
        "Session_Duration_min",
        "Charging_Efficiency_pct",
        "Connector_Temp_C",
        "Grid_Load_kW",
    }
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["Computed_Power_kW"] = data["Voltage_V"] * data["Current_A"] / 1000.0

    computed_power = data["Computed_Power_kW"].replace(0, np.nan)
    data["Power_Mismatch_pct"] = (
        (data["Grid_Load_kW"] - computed_power).abs() / computed_power * 100.0
    ).replace([np.inf, -np.inf], np.nan)

    data["Connector_Temp_Rise_C"] = data["Connector_Temp_C"] - data["Temperature_C"]

    session_minutes = data["Session_Duration_min"].replace(0, np.nan)
    data["Energy_Rate_kWh_per_min"] = (
        data["Session_Energy_kWh"] / session_minutes
    ).replace([np.inf, -np.inf], np.nan)

    max_power = data["MaxPowerKW"].replace(0, np.nan)
    data["Power_Utilization_pct"] = (
        data["Computed_Power_kW"] / max_power * 100.0
    ).replace([np.inf, -np.inf], np.nan)

    return data


def add_fault_labels(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["FaultLabel"] = np.nan
    data.loc[data["StatusType"].isin(NORMAL_STATUSES), "FaultLabel"] = 0
    data.loc[data["StatusType"].isin(FAULT_STATUSES), "FaultLabel"] = 1
    return data


def build_model() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=25)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=180,
        max_depth=18,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_model(df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, pd.Series, pd.Series]:
    labelled = df.dropna(subset=["FaultLabel"]).copy()
    if labelled["FaultLabel"].nunique() < 2:
        raise ValueError("Training requires both normal and faulty StatusType records.")

    x = labelled[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = labelled["FaultLabel"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_model()
    model.fit(x_train, y_train)
    return model, x_test, y_test, labelled["StatusType"]


def sensor_rule_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []

    status = row["StatusType"]
    if status in FAULT_STATUSES:
        reasons.append(f"status is {status}")
    elif status == PLANNED_STATUS:
        reasons.append("station is planned for a future date, not an active charger")

    if row["Voltage_V"] < 200:
        reasons.append(f"low voltage: {row['Voltage_V']:.1f} V")

    if row["Current_A"] <= 0 and row["Session_Energy_kWh"] > 2:
        reasons.append("zero current while session energy is recorded")

    if row["MaxPowerKW"] <= 0 and row["Grid_Load_kW"] > 1:
        reasons.append("rated max power is zero while grid load is present")

    if row["Connector_Temp_C"] >= 115:
        reasons.append(f"very high connector temperature: {row['Connector_Temp_C']:.1f} C")
    elif row["Connector_Temp_C"] >= 85:
        reasons.append(f"high connector temperature: {row['Connector_Temp_C']:.1f} C")

    if row["Charging_Efficiency_pct"] < 90:
        reasons.append(f"low charging efficiency: {row['Charging_Efficiency_pct']:.1f}%")

    mismatch = row.get("Power_Mismatch_pct", np.nan)
    if pd.notna(mismatch) and mismatch > 35:
        reasons.append(f"grid load and computed power mismatch: {mismatch:.1f}%")

    utilization = row.get("Power_Utilization_pct", np.nan)
    if pd.notna(utilization) and utilization > 125:
        reasons.append(f"power utilization above rating: {utilization:.1f}%")

    return reasons


def rule_based_status(reasons: list[str]) -> str:
    severe_markers = (
        "status is Not Operational",
        "low voltage",
        "zero current",
    )
    if any(reason.startswith(severe_markers) for reason in reasons):
        return "CRITICAL"
    if reasons:
        return "WARNING"
    return "NORMAL"


def known_status_label(status: str) -> str:
    if status in FAULT_STATUSES:
        return "FAULT"
    if status in NORMAL_STATUSES:
        return "NORMAL"
    if status == PLANNED_STATUS:
        return "PLANNED_FUTURE"
    return "UNKNOWN"


def add_predictions(
    df: pd.DataFrame,
    model: Pipeline,
    fault_threshold: float,
) -> pd.DataFrame:
    results = df.copy()
    features = results[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    fault_probability = model.predict_proba(features)[:, 1]
    ml_prediction = np.where(fault_probability >= fault_threshold, "HIGH_RISK", "NORMAL")

    results["ML_Fault_Probability"] = np.round(fault_probability, 4)
    results["ML_Risk_Status"] = ml_prediction
    results["Known_Status_Label"] = results["StatusType"].apply(known_status_label)

    reasons = results.apply(sensor_rule_reasons, axis=1)
    results["Rule_Reasons"] = reasons.apply(lambda items: "; ".join(items) or "none")
    results["Sensor_Risk_Status"] = reasons.apply(rule_based_status)

    results["Final_Fault_Status"] = np.select(
        [
            results["Known_Status_Label"].eq("PLANNED_FUTURE"),
            results["Known_Status_Label"].eq("FAULT"),
            results["Sensor_Risk_Status"].eq("CRITICAL"),
            results["ML_Risk_Status"].eq("HIGH_RISK"),
            results["Sensor_Risk_Status"].eq("WARNING"),
        ],
        [
            "PLANNED_FUTURE",
            "FAULT",
            "FAULT",
            "WARNING",
            "WARNING",
        ],
        default="NORMAL",
    )

    output_columns = [
        "StationID",
        "Operator",
        "UsageType",
        "ConnectionTypes",
        "StatusType",
        "Voltage_V",
        "Current_A",
        "Session_Energy_kWh",
        "Temperature_C",
        "Connector_Temp_C",
        "Charging_Efficiency_pct",
        "Grid_Load_kW",
        "Computed_Power_kW",
        "Power_Mismatch_pct",
        "Power_Utilization_pct",
        "Known_Status_Label",
        "ML_Fault_Probability",
        "ML_Risk_Status",
        "Sensor_Risk_Status",
        "Final_Fault_Status",
        "Rule_Reasons",
    ]
    return results[output_columns]


def print_status_summary(results: pd.DataFrame) -> None:
    print("\nFinal fault status summary")
    print("=" * 32)
    print(results["Final_Fault_Status"].value_counts().to_string())

    print("\nTop 10 highest risk stations")
    print("=" * 32)
    top_risk = results.sort_values(
        ["ML_Fault_Probability", "Final_Fault_Status"],
        ascending=[False, True],
    ).head(10)
    columns = [
        "StationID",
        "StatusType",
        "ML_Fault_Probability",
        "Final_Fault_Status",
        "Rule_Reasons",
    ]
    print(top_risk[columns].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and run EV charging station fault detection."
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default=DEFAULT_DATASET,
        help=f"Input CSV file. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--results-file",
        default=DEFAULT_RESULTS_FILE,
        help=f"Output predictions CSV. Default: {DEFAULT_RESULTS_FILE}",
    )
    parser.add_argument(
        "--model-file",
        default=DEFAULT_MODEL_FILE,
        help=f"Output trained model file. Default: {DEFAULT_MODEL_FILE}",
    )
    parser.add_argument(
        "--fault-threshold",
        type=float,
        default=0.70,
        help=(
            "ML probability threshold for HIGH_RISK warnings. "
            "Default: 0.70"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.csv_file)
    results_path = Path(args.results_file)
    model_path = Path(args.model_file)

    df = load_dataset(dataset_path)
    df = add_engineered_features(df)
    df = add_fault_labels(df)

    model, x_test, y_test, _ = train_model(df)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= args.fault_threshold).astype(int)

    print("Model evaluation on held-out test data")
    print("=" * 40)
    print(f"ML high-risk threshold: {args.fault_threshold:.2f}")
    print(classification_report(y_test, predictions, target_names=["NORMAL", "FAULT"]))
    print("Confusion matrix [[normal, false fault], [missed fault, fault]]")
    print(confusion_matrix(y_test, predictions))

    results = add_predictions(df, model, args.fault_threshold)
    results.to_csv(results_path, index=False)
    joblib.dump(model, model_path)

    print_status_summary(results)
    print(f"\nSaved predictions to: {results_path.resolve()}")
    print(f"Saved trained model to: {model_path.resolve()}")


if __name__ == "__main__":
    main()
