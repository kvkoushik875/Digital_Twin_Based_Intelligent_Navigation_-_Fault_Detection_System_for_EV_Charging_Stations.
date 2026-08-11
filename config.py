from pathlib import Path

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
LATENT_DIM = 8

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"

MODEL_PATH = MODEL_DIR / "autoencoder.pt"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
METADATA_PATH = MODEL_DIR / "metadata.pkl"
RESULTS_PATH = MODEL_DIR / "anomaly_results.csv"

COMMUNICATION_TIMEOUT_MINUTES = 15