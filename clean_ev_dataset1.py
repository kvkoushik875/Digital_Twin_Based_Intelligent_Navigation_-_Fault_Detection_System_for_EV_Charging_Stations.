import pandas as pd
import numpy as np
import unicodedata
import re

# ============================================================
# EV CHARGING STATION — DATA CLEANING SCRIPT
# Steps:
#   1. Load raw dataset
#   2. Drop Operator and UsageType columns
#   3. Rename all columns to new names
#   4. Clean MaxPowerKW outliers (cap at 350 kW)
#   5. Clean ConnectionTypes — remove unwanted symbols
#   6. Clean StatusType — standardise values
#   7. Save cleaned dataset
# ============================================================

INPUT_PATH  = r"ev_charging_full_sensor_dataset.csv"
OUTPUT_PATH = r"ev_charging_cleaned.csv"

# ------------------------------------------------------------------
# STEP 1: LOAD
# ------------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv(INPUT_PATH)
print(f"  Original shape : {df.shape}")
print(f"  Columns        : {df.columns.tolist()}")


# ------------------------------------------------------------------
# STEP 2: DROP Operator and UsageType columns
# ------------------------------------------------------------------
print("\n[STEP 2] Dropping columns: Operator, UsageType, StatusType")
df.drop(columns=['Operator', 'UsageType', 'StatusType'], inplace=True)
print(f"  Shape after drop : {df.shape}")


# ------------------------------------------------------------------
# STEP 3: RENAME columns
# Old name               → New name
# StationID              → StationID          (kept as-is)
# MaxPowerKW             → maximum_power_kw
# FastChargeCount        → fast_charging_connector_type
# ConnectionTypes        → connector_standard
# StatusType             → charger_status_type
# Voltage_V              → supply_voltage_v
# Current_A              → output_current_a
# Session_Energy_kWh     → session_energy_consumed_kwh
# Temperature_C          → charger_temperature_celsius
# SoC_pct                → battery_soc_percent
# Session_Duration_min   → charging_session_duration_min
# Charging_Efficiency_pct→ charging_state
# Connector_Temp_C       → connector_standard  ← NOTE: mapped below
# Grid_Load_kW           → grid_power_load_kw
# ------------------------------------------------------------------
print("\n[STEP 3] Renaming columns...")
rename_map = {
    'MaxPowerKW'             : 'maximum_power_kw',
    'FastChargeCount'        : 'fast_charging_connector_type',
    'ConnectionTypes'        : 'connector_standard',
    'Voltage_V'              : 'supply_voltage_v',
    'Current_A'              : 'output_current_a',
    'Session_Energy_kWh'     : 'session_energy_consumed_kwh',
    'Temperature_C'          : 'charger_temperature_celsius',
    'SoC_pct'                : 'battery_soc_percent',
    'Session_Duration_min'   : 'charging_session_duration_min',
    'Charging_Efficiency_pct': 'charging_state',
    'Connector_Temp_C'       : 'connector_temperature_celsius',
    'Grid_Load_kW'           : 'grid_power_load_kw',
}
df.rename(columns=rename_map, inplace=True)
print(f"  Renamed columns : {df.columns.tolist()}")


# ------------------------------------------------------------------
# STEP 4: CLEAN maximum_power_kw — cap outliers at 350 kW
# Real-world EV chargers max ~350 kW; values above are corrupt data
# ------------------------------------------------------------------
print("\n[STEP 4] Cleaning maximum_power_kw outliers (> 350 kW)...")
outlier_mask = df['maximum_power_kw'] > 350
print(f"  Outlier rows found : {outlier_mask.sum()}")
df.loc[outlier_mask, 'maximum_power_kw'] = 350
print(f"  All values > 350 kW capped to 350 kW")
print(f"  New max value : {df['maximum_power_kw'].max()}")


# ------------------------------------------------------------------
# STEP 5: CLEAN connector_standard — remove unwanted symbols
# Replace semicolons with commas, strip extra whitespace
# ------------------------------------------------------------------
print("\n[STEP 5] Cleaning connector_standard — removing unwanted symbols...")
before = df['connector_standard'].str.contains(';', na=False).sum()
df['connector_standard'] = (
    df['connector_standard']
    .astype(str)
    .str.replace(';', ',', regex=False)
    .str.strip()
)
print(f"  Rows with semicolons fixed : {before}")


# ------------------------------------------------------------------
# STEP 6: ENFORCE FINAL COLUMN ORDER
# StationID first, then all renamed columns in requested order
# ------------------------------------------------------------------
print("\n[STEP 6] Enforcing final column order...")
final_cols = [
    'StationID',
    'maximum_power_kw',
    'fast_charging_connector_type',
    'supply_voltage_v',
    'output_current_a',
    'session_energy_consumed_kwh',
    'charger_temperature_celsius',
    'battery_soc_percent',
    'charging_session_duration_min',
    'charging_state',
    'connector_standard',
    'grid_power_load_kw',
]
df = df[final_cols]


# ------------------------------------------------------------------
# STEP 8: FINAL QUALITY CHECK
# ------------------------------------------------------------------
print("\n[STEP 7] Final quality check...")
print(f"  Shape            : {df.shape}")
print(f"  Total NaN values : {df.isna().sum().sum()}")
print(f"  Duplicate rows   : {df.duplicated().sum()}")
print(f"\n  Column list ({len(df.columns)}):")
for i, c in enumerate(df.columns, 1):
    print(f"    {i:2}. {c}")

print(f"\n  Numeric column stats:")
num_cols = [
    'maximum_power_kw', 'supply_voltage_v', 'output_current_a',
    'session_energy_consumed_kwh', 'charger_temperature_celsius',
    'battery_soc_percent', 'charging_session_duration_min',
    'charging_state', 'grid_power_load_kw'
]
print(df[num_cols].describe().round(2).to_string())

print(f"\n  Sample rows (first 5):")
print(df.head(5).to_string())


# ------------------------------------------------------------------
# STEP 8: SAVE
# ------------------------------------------------------------------
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nCleaned dataset saved → {OUTPUT_PATH}")
print(f"Final shape          : {df.shape}")

