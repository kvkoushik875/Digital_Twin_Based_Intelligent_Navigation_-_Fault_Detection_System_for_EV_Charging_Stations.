"""Rule-based diagnosis of EV charging fault types."""

from typing import Any


BATTERY_TEMP_LIMIT = 45.0
STATION_TEMP_LIMIT = 50.0
THERMAL_RISE_LIMIT = 20.0
EFFICIENCY_MINIMUM = 90.0
DEGRADATION_LIMIT = 15.0
SOC_GAIN_MINIMUM = 5.0


def detect_fault_types(
    sensor_data: dict[str, Any],
) -> list[str]:
    """Return probable fault types from sensor parameters."""

    faults: list[str] = []
    voltage = float(sensor_data["voltage"])
    current = float(sensor_data["current"])
    battery_temp = float(sensor_data["battery_temp"])
    ambient_temp = float(sensor_data["ambient_temp"])
    station_temp = float(sensor_data["temperature"])
    efficiency = float(sensor_data["efficiency"])
    degradation = float(sensor_data["degradation_rate"])
    charging_rate = float(sensor_data["charging_rate_kw"])
    soc_start = float(sensor_data["soc_start"])
    soc_end = float(sensor_data["soc_end"])

    if voltage <= 0:
        faults.append("VOLTAGE_FAILURE")

    if current <= 0:
        faults.append("NO_CHARGING_CURRENT")

    if charging_rate <= 0:
        faults.append("CHARGING_RATE_FAILURE")

    if battery_temp > BATTERY_TEMP_LIMIT:
        faults.append("BATTERY_OVERHEATING")

    if station_temp > STATION_TEMP_LIMIT:
        faults.append("STATION_OVERHEATING")

    if battery_temp - ambient_temp > THERMAL_RISE_LIMIT:
        faults.append("ABNORMAL_THERMAL_RISE")

    if efficiency < EFFICIENCY_MINIMUM:
        faults.append("LOW_CHARGING_EFFICIENCY")

    if degradation > DEGRADATION_LIMIT:
        faults.append("HIGH_BATTERY_DEGRADATION")

    soc_gain = soc_end - soc_start

    if soc_end <= soc_start:
        faults.append("SOC_CHARGING_FAILURE")
    elif soc_gain < SOC_GAIN_MINIMUM:
        faults.append("LOW_SOC_GAIN")

    return faults


if __name__ == "__main__":
    sample = {
        "voltage": 4.11,
        "current": 24.21,
        "battery_temp": 47.0,
        "ambient_temp": 25.0,
        "temperature": 52.0,
        "efficiency": 88.0,
        "degradation_rate": 16.0,
        "charging_rate_kw": 28.5,
        "soc_start": 64.0,
        "soc_end": 67.0,
    }

    print("Detected fault types:")
    print(detect_fault_types(sample))