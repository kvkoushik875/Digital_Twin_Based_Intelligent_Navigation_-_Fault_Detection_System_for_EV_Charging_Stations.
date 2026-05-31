import pandas as pd
import numpy as np

# ============================================================
# EV CHARGING STATION — FULL SENSOR DATASET GENERATOR
# Input  : global_ev_charging_station_sensor_data.csv (7 cols)
# Output : ev_charging_full_sensor_dataset.csv        (17 cols)
# New columns added (9):
#   Voltage_V, Current_A, Session_Energy_kWh,
#   Temperature_C, SoC_pct, Session_Duration_min,
#   Charging_Efficiency_pct, Connector_Temp_C,
#   Grid_Load_kW
# NOTE: Original data is NOT cleaned or modified in any way.
# ============================================================

np.random.seed(42)

# ---- STEP 1: LOAD ORIGINAL DATASET (no cleaning) ----
INPUT_PATH  = r"global_ev_charging_station_sensor_data.csv"
OUTPUT_PATH = r"ev_charging_full_sensor_dataset.csv"

print("Loading dataset...")
df = pd.read_csv(INPUT_PATH)
n  = len(df)
print(f"  Rows    : {n}")
print(f"  Columns : {df.columns.tolist()}")


# ---- STEP 2: SHADOW POWER VARIABLE (original column untouched) ----
# Used only for physics calculations.
# Outlier MaxPowerKW (e.g. 1,000,000) capped at 350 kW for realism.
# Zero-power rows treated as 7 kW slow chargers.
power_kw = df['MaxPowerKW'].clip(upper=350).replace(0, 7.0)


# ---- STEP 3: Voltage_V ----
# AC slow  (<50 kW)  : ~230 V  (single-phase AC)
# DC fast  (50-149)  : ~400 V  (standard DC fast charger)
# Ultra    (>=150)   : ~800 V  (high-power DC ultra-fast)
voltage = np.zeros(n)
mask_slow  = power_kw < 50
mask_fast  = (power_kw >= 50) & (power_kw < 150)
mask_ultra = power_kw >= 150
voltage[mask_slow]  = np.round(np.random.normal(230, 8,  mask_slow.sum()),  1)
voltage[mask_fast]  = np.round(np.random.normal(400, 12, mask_fast.sum()),  1)
voltage[mask_ultra] = np.round(np.random.normal(800, 15, mask_ultra.sum()), 1)
df['Voltage_V'] = np.clip(voltage, 100, 850)


# ---- STEP 4: Current_A ----
# Ohm's law: I = (P_kW * 1000) / V  +  Gaussian noise (sigma=3 A)
current_raw = (power_kw * 1000 / df['Voltage_V']) + np.random.normal(0, 3, n)
df['Current_A'] = np.round(np.clip(current_raw, 0, 500), 2)


# ---- STEP 5: Session_Energy_kWh ----
# Typical delivered energy scales with charger speed tier:
#   slow (<50 kW)  -> mean 18 kWh
#   fast (50-149)  -> mean 35 kWh
#   ultra (>=150)  -> mean 55 kWh
# Standard deviation = 25% of mean; range clamped 2-120 kWh
session_mean = np.where(power_kw >= 150, 55,
               np.where(power_kw >= 50,  35, 18))
df['Session_Energy_kWh'] = np.round(
    np.clip(np.random.normal(session_mean, session_mean * 0.25, n), 2, 120), 2
)


# ---- STEP 6: Temperature_C ----
# Ambient base 22 degC (sigma=6) + load-proportional heating (up to +8 degC)
load_factor = power_kw / power_kw.max()
df['Temperature_C'] = np.round(
    np.clip(np.random.normal(22, 6, n) + load_factor * 8, -5, 60), 1
)


# ---- STEP 7: SoC_pct ----
# State of Charge at session start: normal distribution, range 5-99%
df['SoC_pct'] = np.round(
    np.clip(np.random.normal(45, 20, n), 5, 99), 1
)


# ---- STEP 8: Session_Duration_min ----
# Duration (min) = (Energy_kWh / Power_kW) * 60  *  random variation +-15%
# Clamped 5-480 minutes
duration_base = (df['Session_Energy_kWh'] / power_kw) * 60
df['Session_Duration_min'] = np.round(
    np.clip(duration_base * np.random.uniform(0.85, 1.15, n), 5, 480), 1
)


# ---- STEP 9: Charging_Efficiency_pct ----
# DC conversion losses increase with power level:
#   ultra (>=150 kW) : 92%  (higher switching losses)
#   fast  (50-149)   : 94%
#   slow  (<50 kW)   : 96%
base_eff = np.where(power_kw >= 150, 92,
           np.where(power_kw >= 50,  94, 96))
df['Charging_Efficiency_pct'] = np.round(
    np.clip(np.random.normal(base_eff, 1.5, n), 80, 99.5), 1
)


# ---- STEP 10: Connector_Temp_C ----
# Joule heating model (same formula as EV fault detection script):
#   T_plug = T_ambient + 10 + (I^2 * R_contact)
#   R_contact = 0.015 Ohm  (standard connector pin resistance)
# Clamped at 120 degC (connector material thermal limit)
df['Connector_Temp_C'] = np.round(
    (df['Temperature_C'] + 10 + (df['Current_A'] ** 2 * 0.015)), 1
).clip(upper=120)


# ---- STEP 11: Grid_Load_kW ----
# Actual grid draw = station power / charging efficiency + background noise
df['Grid_Load_kW'] = np.round(
    power_kw * df['Charging_Efficiency_pct'] / 100
    + np.random.normal(0, power_kw * 0.05, n), 2
).clip(lower=0)


# ---- STEP 12: ENFORCE FINAL COLUMN ORDER ----
final_cols = [
    'StationID', 'Operator', 'UsageType', 'MaxPowerKW', 'FastChargeCount',
    'ConnectionTypes', 'StatusType',
    'Voltage_V', 'Current_A', 'Session_Energy_kWh', 'Temperature_C',
    'SoC_pct', 'Session_Duration_min', 'Charging_Efficiency_pct',
    'Connector_Temp_C', 'Grid_Load_kW'
]
df = df[final_cols]


# ---- STEP 13: SUMMARY REPORT ----
print(f"\n========== GENERATION COMPLETE ==========")
print(f"Final shape  : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumns ({len(df.columns)}):")
for i, c in enumerate(df.columns, 1):
    print(f"  {i:2}. {c}")

sensor_cols = [
    'Voltage_V', 'Current_A', 'Session_Energy_kWh', 'Temperature_C',
    'SoC_pct', 'Session_Duration_min', 'Charging_Efficiency_pct',
    'Connector_Temp_C', 'Grid_Load_kW'
]
print("\nSensor statistics:")
print(df[sensor_cols].describe().round(2).to_string())

print("\nSample output (first 5 rows):")
print(df.head(5).to_string())


# ---- STEP 14: SAVE ----
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nDataset saved -> {OUTPUT_PATH}")
