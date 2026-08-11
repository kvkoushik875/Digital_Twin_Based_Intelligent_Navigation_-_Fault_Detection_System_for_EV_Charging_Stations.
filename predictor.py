"""Inference engine for EV charging fault detection."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from ai_engine.autoencoder import EVAutoencoder


# =========================================================
# Saved artifact paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"

MODEL_PATH = MODEL_DIR / "autoencoder.pt"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
METADATA_PATH = MODEL_DIR / "metadata.pkl"


class FaultPredictor:
    """Detect anomalies using the trained Autoencoder."""

    def __init__(self) -> None:
        self._check_artifacts()

        self.metadata = joblib.load(METADATA_PATH)
        self.scaler = joblib.load(SCALER_PATH)

        self.features: list[str] = self.metadata["features"]

        self.warning_threshold = float(
            self.metadata["warning_threshold"]
        )
        self.critical_threshold = float(
            self.metadata["critical_threshold"]
        )
        self.failure_threshold = float(
            self.metadata["failure_threshold"]
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = EVAutoencoder(
            input_dim=int(self.metadata["input_dim"]),
            latent_dim=int(self.metadata["latent_dim"]),
        ).to(self.device)

        state_dict = torch.load(
            MODEL_PATH,
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _check_artifacts(self) -> None:
        """Check whether all trained-model files exist."""

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
                + ". Run: python -m ai_engine.train"
            )

    def _prepare_sensor_data(
        self,
        sensor_data: dict[str, Any],
    ) -> pd.DataFrame:
        """Validate and arrange features in training order."""

        missing_features = [
            feature
            for feature in self.features
            if feature not in sensor_data
            or sensor_data[feature] is None
        ]

        if missing_features:
            raise ValueError(
                f"Missing sensor features: {missing_features}"
            )

        prepared_data: dict[str, float] = {}

        for feature in self.features:
            try:
                value = float(sensor_data[feature])

            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"'{feature}' must contain a numerical value."
                ) from error

            if not np.isfinite(value):
                raise ValueError(
                    f"'{feature}' contains an invalid value."
                )

            prepared_data[feature] = value

        return pd.DataFrame(
            [prepared_data],
            columns=self.features,
        )

    def classify_fault(
        self,
        anomaly_score: float,
    ) -> str:
        """Convert anomaly score into fault severity."""

        if anomaly_score >= self.failure_threshold:
            return "FAILURE"

        if anomaly_score >= self.critical_threshold:
            return "CRITICAL"

        if anomaly_score >= self.warning_threshold:
            return "WARNING"

        return "NORMAL"

    def predict(
        self,
        sensor_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate anomaly score and fault severity."""

        dataframe = self._prepare_sensor_data(
            sensor_data
        )

        scaled_data = self.scaler.transform(
            dataframe
        )

        input_tensor = torch.tensor(
            scaled_data,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.inference_mode():
            reconstructed = self.model(
                input_tensor
            )

            anomaly_score = torch.mean(
                torch.square(
                    input_tensor - reconstructed
                ),
                dim=1,
            ).item()

        severity = self.classify_fault(
            anomaly_score
        )

        return {
            "anomaly_score": round(
                anomaly_score,
                8,
            ),
            "fault_severity": severity,
            "is_anomaly": severity != "NORMAL",
        }


# =========================================================
# Local predictor test
# =========================================================

if __name__ == "__main__":
    predictor = FaultPredictor()

    sample = {
        "soc_percent": 38.2920383,
        "voltage": 4.118650119,
        "current": 24.21593406,
        "battery_temp": 38.29097328,
        "ambient_temp": 25.96290405,
        "degradation_rate": 10.25337603,
        "efficiency": 97.94932479,
        "charging_cycles": 172,
        "battery_capacity_kwh": 85.0,
        "energy_consumed_kwh": 29.16952954,
        "charging_rate_kw": 28.50456454,
        "soc_start": 64.06837901,
        "soc_end": 83.78854261,
        "temperature": 42.73966105,
    }

    result = predictor.predict(sample)

    print("\nFault detection result")
    print(result)