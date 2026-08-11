"""Diagnostic fault engine built on the Autoencoder predictor."""

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from ai_engine.predictor import FaultPredictor


COMMUNICATION_TIMEOUT_MINUTES = 15


class FaultDetectionEngine:
    """Combine Autoencoder detection with parameter-based diagnosis."""

    def __init__(self) -> None:
        self.predictor = FaultPredictor()

    def _has_communication_failure(
        self,
        sensor_data: dict[str, Any],
    ) -> bool:
        """Check whether sensor updates have stopped."""

        recorded_at = sensor_data.get("recorded_at")

        # Communication status cannot be evaluated
        # without a timestamp.
        if recorded_at is None:
            return False

        timestamp = pd.to_datetime(
            recorded_at,
            utc=True,
            errors="coerce",
        )

        if pd.isna(timestamp):
            return True

        data_age = (
            datetime.now(timezone.utc)
            - timestamp.to_pydatetime()
        )

        return data_age > timedelta(
            minutes=COMMUNICATION_TIMEOUT_MINUTES
        )

    def identify_fault_types(
        self,
        sensor_data: dict[str, Any],
    ) -> list:
        """Suggest likely fault causes using parameter rules."""

        faults: list[str] = []

        voltage = float(sensor_data["voltage"])
        current = float(sensor_data["current"])
        battery_temp = float(sensor_data["battery_temp"])
        ambient_temp = float(sensor_data["ambient_temp"])
        temperature = float(sensor_data["temperature"])
        efficiency = float(sensor_data["efficiency"])
        degradation_rate = float(
            sensor_data["degradation_rate"]
        )
        charging_rate = float(
            sensor_data["charging_rate_kw"]
        )
        soc_start = float(sensor_data["soc_start"])
        soc_end = float(sensor_data["soc_end"])

        if voltage <= 0:
            faults.append("VOLTAGE_FAILURE")

        if current <= 0:
            faults.append("NO_CHARGING_CURRENT")

        if charging_rate <= 0:
            faults.append("CHARGING_RATE_FAILURE")

        if battery_temp > 45:
            faults.append("BATTERY_OVERHEATING")

        if temperature > 50:
            faults.append("STATION_OVERHEATING")

        if battery_temp - ambient_temp > 20:
            faults.append("ABNORMAL_THERMAL_RISE")

        if efficiency < 90:
            faults.append("LOW_CHARGING_EFFICIENCY")

        if degradation_rate > 15:
            faults.append("HIGH_BATTERY_DEGRADATION")

        if soc_end <= soc_start:
            faults.append("SOC_CHARGING_FAILURE")

        elif soc_end - soc_start < 5:
            faults.append("LOW_SOC_GAIN")

        if self._has_communication_failure(sensor_data):
            faults.append("COMMUNICATION_FAILURE")

        return faults

    def detect(
        self,
        sensor_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Run Autoencoder detection and parameter diagnosis."""

        prediction = self.predictor.predict(
            sensor_data
        )

        fault_types: list[str] = []

        if prediction["is_anomaly"]:
            fault_types = self.identify_fault_types(
                sensor_data
            )

            if not fault_types:
                fault_types = [
                    "MULTIVARIATE_SENSOR_ANOMALY"
                ]

        return {
            "anomaly_score": prediction["anomaly_score"],
            "fault_severity": prediction[
                "fault_severity"
            ],
            "is_anomaly": prediction["is_anomaly"],
            "fault_types": (
                fault_types
                if fault_types
                else ["NO_FAULT"]
            ),
        }


if __name__ == "__main__":
    fault_engine = FaultDetectionEngine()

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

    result = fault_engine.detect(sample)

    print("\nComplete fault detection result")
    print(result)
