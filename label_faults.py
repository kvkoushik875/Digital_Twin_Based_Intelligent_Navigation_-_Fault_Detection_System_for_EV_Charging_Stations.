import pandas as pd

def label_faults(df):
    # Possible column name variations
    COLUMN_MAP = {
        "voltage": ["voltage", "Voltage", "Voltage_V", "V"],
        "current": ["current", "Current", "Current_A", "A"],
        "temperature": ["temperature", "Temperature", "Temp", "Temp_C"],
        "power": ["power", "Power", "Power_kW"],
        "soc": ["soc", "SOC", "SOC_percent"]
    }

    # Resolve actual column names
    resolved = {}

    for key, options in COLUMN_MAP.items():
        for col in options:
            if col in df.columns:
                resolved[key] = col
                break

    print("Resolved columns:", resolved)

    # Create fault_type column
    df["fault_type"] = "normal"

    # Voltage fluctuation
    if "voltage" in resolved:
        df.loc[df[resolved["voltage"]].diff().abs() > 15, "fault_type"] = "voltage_fluctuation"

    # Current spike
    if "current" in resolved:
        df.loc[df[resolved["current"]] > df[resolved["current"]].mean() + 3 * df[resolved["current"]].std(),
               "fault_type"] = "current_spike"

    # Overheating
    if "temperature" in resolved:
        df.loc[df[resolved["temperature"]] > 60, "fault_type"] = "overheat"

    # Low SOC
    if "soc" in resolved:
        df.loc[df[resolved["soc"]] < 10, "fault_type"] = "low_soc"

    return df
