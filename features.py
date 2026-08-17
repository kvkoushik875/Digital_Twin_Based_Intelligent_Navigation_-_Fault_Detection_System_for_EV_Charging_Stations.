FEATURE_COLUMNS = [
    "soc_percent",
    "voltage",
    "current",
    "battery_temp",
    "ambient_temp",
    "charging_duration_min",
    "degradation_rate",
    "efficiency",
    "charging_cycles",
    "battery_capacity_kwh",
    "energy_consumed_kwh",
    "charging_duration_hours",
    "charging_rate_kw",
    "soc_start",
    "soc_end",
    "temperature",
]

# Used only to give unsupervised fault clusters (see train.py) a
# human-readable name from their dominant feature - maps to 3 of the 5
# fault categories named in the project abstract (voltage instability,
# excessive power loss, charger overheating). Communication failures and
# interrupted charging sessions have no corresponding column in
# sensor_data, so they aren't detectable here yet.
FEATURE_FAULT_LABELS = {
    "soc_percent": "SOC_ANOMALY",
    "voltage": "VOLTAGE_INSTABILITY",
    "current": "CURRENT_ANOMALY",
    "battery_temp": "CHARGER_OVERHEATING",
    "ambient_temp": "CHARGER_OVERHEATING",
    "charging_duration_min": "CHARGING_DURATION_ANOMALY",
    "degradation_rate": "DEGRADATION_ANOMALY",
    "efficiency": "EXCESSIVE_POWER_LOSS",
    "charging_cycles": "CHARGING_CYCLES_ANOMALY",
    "battery_capacity_kwh": "BATTERY_CAPACITY_ANOMALY",
    "energy_consumed_kwh": "EXCESSIVE_POWER_LOSS",
    "charging_duration_hours": "CHARGING_DURATION_ANOMALY",
    "charging_rate_kw": "EXCESSIVE_POWER_LOSS",
    "soc_start": "SOC_ANOMALY",
    "soc_end": "SOC_ANOMALY",
    "temperature": "CHARGER_OVERHEATING",
}
