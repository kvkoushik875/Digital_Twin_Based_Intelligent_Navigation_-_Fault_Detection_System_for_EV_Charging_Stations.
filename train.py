"""Train an unsupervised EV charging Autoencoder."""

from pathlib import Path
import random

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sqlalchemy import text
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ai_engine.autoencoder import EVAutoencoder
from database import engine


# =========================================================
# Configuration
# =========================================================

RANDOM_SEED = 42
LATENT_DIM = 8

EPOCHS = 150
BATCH_SIZE = 256
LEARNING_RATE = 0.001
VALIDATION_SIZE = 0.20

EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 1e-6


# =========================================================
# Saved model paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"

MODEL_PATH = MODEL_DIR / "autoencoder.pt"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
METADATA_PATH = MODEL_DIR / "metadata.pkl"


# =========================================================
# Model input features
# =========================================================

FEATURES = [
    "soc_percent",
    "voltage",
    "current",
    "battery_temp",
    "ambient_temp",
    "degradation_rate",
    "efficiency",
    "charging_cycles",
    "battery_capacity_kwh",
    "energy_consumed_kwh",
    "charging_rate_kw",
    "soc_start",
    "soc_end",
    "temperature",
]

INPUT_DIM = len(FEATURES)


# =========================================================
# PostgreSQL query
# =========================================================

SENSOR_QUERY = """
SELECT
    soc_percent,
    voltage,
    current,
    battery_temp,
    ambient_temp,
    degradation_rate,
    efficiency,
    charging_cycles,
    battery_capacity_kwh,
    energy_consumed_kwh,
    charging_rate_kw,
    soc_start,
    soc_end,
    temperature
FROM sensor_data
"""


# =========================================================
# Utility functions
# =========================================================

def set_random_seed() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def load_sensor_data() -> pd.DataFrame:
    """Load every sensor_data row from PostgreSQL."""

    print("Loading sensor data from PostgreSQL...")

    with engine.connect() as connection:
        database_count = connection.execute(
            text("SELECT COUNT(*) FROM sensor_data")
        ).scalar_one()

        dataframe = pd.read_sql_query(
            sql=text(SENSOR_QUERY),
            con=connection,
        )

    print(f"Rows in PostgreSQL: {database_count}")
    print(f"Rows loaded into Pandas: {len(dataframe)}")

    if dataframe.empty:
        raise RuntimeError(
            "The sensor_data table contains no records."
        )

    if len(dataframe) != database_count:
        raise RuntimeError(
            f"Database contains {database_count} rows, "
            f"but Pandas loaded {len(dataframe)} rows."
        )

    missing_columns = [
        feature
        for feature in FEATURES
        if feature not in dataframe.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f"Missing database columns: {missing_columns}"
        )

    # The user requested no data-cleaning stage.
    # Fail clearly if unsuitable values are present.
    if dataframe[FEATURES].isnull().any().any():
        null_counts = dataframe[FEATURES].isnull().sum()

        raise RuntimeError(
            "NULL values detected:\n"
            f"{null_counts[null_counts > 0]}"
        )

    if not all(
        pd.api.types.is_numeric_dtype(dataframe[column])
        for column in FEATURES
    ):
        non_numeric = [
            column
            for column in FEATURES
            if not pd.api.types.is_numeric_dtype(
                dataframe[column]
            )
        ]

        raise RuntimeError(
            f"Non-numeric feature columns: {non_numeric}"
        )

    values = dataframe[FEATURES].to_numpy()

    if not np.isfinite(values).all():
        raise RuntimeError(
            "Infinite or invalid numerical values detected."
        )

    return dataframe


def calculate_errors(
    model: EVAutoencoder,
    tensor: torch.Tensor,
) -> np.ndarray:
    """Calculate one reconstruction error per record."""

    model.eval()

    with torch.inference_mode():
        reconstructed = model(tensor)

        errors = torch.mean(
            torch.square(tensor - reconstructed),
            dim=1,
        )

    return errors.detach().cpu().numpy()


# =========================================================
# Training
# =========================================================

def train() -> None:
    set_random_seed()

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training device: {device}")

    dataframe = load_sensor_data()

    total_records = len(dataframe)

    train_frame, validation_frame = train_test_split(
        dataframe,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    print(f"Total records used: {total_records}")
    print(f"Training records: {len(train_frame)}")
    print(f"Validation records: {len(validation_frame)}")

    # Fit the scaler only on training records.
    scaler = RobustScaler()

    train_scaled = scaler.fit_transform(
        train_frame[FEATURES]
    )

    validation_scaled = scaler.transform(
        validation_frame[FEATURES]
    )

    train_tensor = torch.tensor(
        train_scaled,
        dtype=torch.float32,
    )

    validation_tensor = torch.tensor(
        validation_scaled,
        dtype=torch.float32,
        device=device,
    )

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    model = EVAutoencoder(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
    ).to(device)

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5,
    )

    best_validation_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print("\nAutoencoder training started\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_training_loss = 0.0

        for (batch,) in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            reconstructed = model(batch)

            loss = loss_function(
                reconstructed,
                batch,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_training_loss += (
                loss.item() * batch.size(0)
            )

        training_loss = (
            total_training_loss / len(train_tensor)
        )

        model.eval()

        with torch.inference_mode():
            validation_output = model(
                validation_tensor
            )

            validation_loss = loss_function(
                validation_output,
                validation_tensor,
            ).item()

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train: {training_loss:.8f} | "
            f"Validation: {validation_loss:.8f}",
            flush=True,
        )

        improvement = (
            best_validation_loss - validation_loss
        )

        if improvement > MIN_DELTA:
            best_validation_loss = validation_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                model.state_dict(),
                MODEL_PATH,
            )

            print("Best model saved.", flush=True)

        else:
            patience_counter += 1

            print(
                "No significant improvement: "
                f"{patience_counter}/"
                f"{EARLY_STOPPING_PATIENCE}",
                flush=True,
            )

        if (
            patience_counter
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                f"\nEarly stopping at epoch {epoch}.",
                flush=True,
            )
            break

    # Restore the best epoch.
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True,
        )
    )

    validation_errors = calculate_errors(
        model,
        validation_tensor,
    )

    warning_threshold = float(
        np.quantile(validation_errors, 0.95)
    )

    critical_threshold = float(
        np.quantile(validation_errors, 0.99)
    )

    failure_threshold = float(
        np.quantile(validation_errors, 0.995)
    )

    joblib.dump(
        scaler,
        SCALER_PATH,
    )

    metadata = {
        "features": FEATURES,
        "input_dim": INPUT_DIM,
        "latent_dim": LATENT_DIM,
        "warning_threshold": warning_threshold,
        "critical_threshold": critical_threshold,
        "failure_threshold": failure_threshold,
        "best_validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "database_records": total_records,
        "training_records": len(train_frame),
        "validation_records": len(validation_frame),
        "batch_size": BATCH_SIZE,
        "maximum_epochs": EPOCHS,
    }

    joblib.dump(
        metadata,
        METADATA_PATH,
    )

    print("\nTraining completed successfully.", flush=True)

    print("\nTraining summary")
    print(f"Database records: {total_records}")
    print(f"Training records: {len(train_frame)}")
    print(f"Validation records: {len(validation_frame)}")
    print(f"Best epoch: {best_epoch}")
    print(
        f"Best validation loss: "
        f"{best_validation_loss:.8f}"
    )

    print("\nAnomaly thresholds")
    print(f"Warning:  {warning_threshold:.8f}")
    print(f"Critical: {critical_threshold:.8f}")
    print(f"Failure:  {failure_threshold:.8f}")

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Scaler saved: {SCALER_PATH}")
    print(f"Metadata saved: {METADATA_PATH}")


if __name__ == "__main__":
    print("Starting AI training...", flush=True)
    train()