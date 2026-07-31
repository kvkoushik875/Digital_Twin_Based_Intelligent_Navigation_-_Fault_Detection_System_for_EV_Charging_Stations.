# EV Battery Failure Prediction Dataset — Feature Descriptions

**Rows:** 200,000 &nbsp;|&nbsp; **Columns:** 70 &nbsp;|&nbsp; **Target:** `battery_failure` (binary) &nbsp;|&nbsp; **Failure rate:** ~10%

This is a synthetic dataset. Every feature is generated from underlying battery-degradation
physics rather than independent random sampling, so the correlations between features (and
between features and the target) mirror real-world electric vehicle battery behavior:
chemistry-specific aging curves, thermal stress, charging habits, driving behavior, and
maintenance quality all interact to determine failure risk.

---

## 1. Vehicle Information
| Column | Description |
|---|---|
| `vehicle_id` | Unique identifier for each vehicle record. |
| `vehicle_brand` | Manufacturer/brand of the EV (20 brands, realistically weighted by market presence). |
| `vehicle_model` | Specific model name within the brand. |
| `vehicle_type` | Body style: Sedan, SUV, Hatchback, Crossover, Truck, or Van. |
| `manufacturing_year` | Year the vehicle was manufactured (2015–2024). |
| `battery_manufacturer` | Company that manufactured the battery pack/cells (CATL, LG Energy Solution, Panasonic, etc.). |
| `battery_chemistry` | Cell chemistry: LFP, NMC, NCA, LMO, or LTO — each with distinct degradation and thermal-risk profiles. |
| `battery_capacity_kwh` | Nominal usable battery capacity, correlated with vehicle type (trucks/SUVs > sedans/hatchbacks). |
| `drive_type` | Drivetrain: FWD, RWD, or AWD. |
| `odometer_km` | Total distance driven, built up from vehicle age and daily usage. |
| `vehicle_age_years` | Age of the vehicle, derived from manufacturing year. |
| `fleet_or_private` | Ownership type. Fleet vehicles see higher mileage, more fast charging, and more aggressive driving. |

## 2. Battery Information
| Column | Description |
|---|---|
| `battery_serial` | Unique battery pack serial number. |
| `cycle_count` | Total charge/discharge cycles, driven by odometer and capacity. |
| `battery_health_percent` | Overall health proxy (100 − capacity loss), the core degradation output. |
| `state_of_charge` | Current charge level (%). |
| `depth_of_discharge` | Average % of capacity discharged per cycle; rises with aggressive driving. |
| `state_of_health` | SoH (%), closely tracks `battery_health_percent` with independent sensor noise. |
| `cell_voltage_avg` | Average per-cell voltage (V), a function of chemistry's nominal voltage and state of charge. |
| `cell_voltage_std` | Std. dev. of cell voltages — rises with aging and prior faults (cell imbalance signal). |
| `pack_voltage` | Total pack voltage, derived from cell voltage and estimated cell count. |
| `cell_temperature_avg` / `cell_temperature_max` | Operating cell temperatures, driven by ambient conditions, driving aggression, and fast charging. |
| `internal_resistance` | Rises with capacity loss, temperature, and chemistry base resistance. |
| `charge_efficiency` / `discharge_efficiency` | Round-trip efficiencies; degrade with capacity loss and temperature extremes. |
| `remaining_capacity` | Usable kWh today (`battery_capacity_kwh` × health%). |
| `capacity_loss_percent` | % capacity lost to aging — the central degradation driver of the whole dataset. |

## 3. Charging History
| Column | Description |
|---|---|
| `charging_cycles_last_month` | Charging sessions in the last 30 days, tied to daily distance and capacity. |
| `fast_charge_ratio` / `slow_charge_ratio` | Proportion of DC fast vs. AC slow charging sessions (complementary). |
| `average_charge_power_kw` | Average charging power; scales with fast-charge ratio. |
| `average_charging_time` | Average session duration (minutes). |
| `overnight_charging_ratio` / `home_charging_ratio` | Home/private owners skew toward overnight, home charging; fleets skew toward public fast charging. |
| `charging_interruptions` | Count of interrupted sessions — more common with public charging reliance. |
| `overcharge_events` | Count of overcharge events — more common with heavy fast-charge usage. |

## 4. Driving Behavior
| Column | Description |
|---|---|
| `average_speed` | Average driving speed (km/h). |
| `average_trip_distance` | Average per-trip distance (km). |
| `aggressive_acceleration_score` / `hard_braking_score` | Composite 0–100 harsh-driving scores; correlated with each other and with fleet use. |
| `regenerative_braking_usage` | % utilization of regen braking — inversely related to aggressive driving. |
| `highway_driving_ratio` / `city_driving_ratio` | Complementary proportions of driving context. |
| `daily_distance` | Average km driven per day — a key driver of cycle accumulation. |

## 5. Environment
| Column | Description |
|---|---|
| `average_ambient_temperature` / `maximum_temperature` / `minimum_temperature` | Temperature exposure, drawn from a latent per-vehicle climate zone. |
| `humidity` | Average relative humidity (%). |
| `altitude` | Typical operating elevation (m). |
| `terrain_type` | Flat, Hilly, Mountainous, Coastal, or Desert. |
| `dust_exposure` | 0–100 particulate exposure score, elevated in desert terrain. |

## 6. Maintenance
| Column | Description |
|---|---|
| `last_service_days` | Days since last service visit. |
| `cooling_system_health` | 0–100 thermal-management system condition; declines with age and heat exposure. |
| `firmware_updates` | Count of BMS/vehicle firmware updates — brand-dependent (OTA-heavy brands update more often). |
| `previous_faults` | Count of prior logged battery fault codes. |
| `maintenance_score` | Composite 0–100 overall maintenance-quality score. |

## 7. Diagnostic Sensors
| Column | Description |
|---|---|
| `thermal_runaway_risk` | Composite 0–100 risk score; chemistry base risk + capacity loss + resistance + voltage imbalance + peak temperature. |
| `voltage_imbalance` | 0–100 inter-cell imbalance score. |
| `temperature_variance` | Spread between peak cell temperature and ambient temperature. |
| `sensor_fault_count` | Count of sensor malfunctions detected. |
| `BMS_warning_count` | Count of BMS warning flags — rises with thermal risk and voltage imbalance. |
| `abnormal_voltage_events` | Count of out-of-range voltage events. |

## 8. Derived Features
| Column | Description |
|---|---|
| `battery_stress_index` | Composite cumulative stress score (thermal + DoD + fast charging + aggressive driving). |
| `aging_score` | Composite score combining cycle usage ratio, vehicle age, and capacity loss. |
| `thermal_health_score` | Higher-is-better inverse of thermal risk and temperature variance. |
| `charging_quality_score` | Higher-is-better score for good charging habits (slow charging, efficiency, low interruptions). |
| `driving_stress_score` | Composite score of driving-pattern-induced battery stress. |
| `predicted_remaining_life_cycles` | Estimated cycles remaining before end-of-life, scaled by chemistry's max cycle life and current health. |

## 9. Target
| Column | Description |
|---|---|
| `battery_failure` | **1 = failure, 0 = healthy.** Generated from a nonlinear logistic combination of ~15 risk/protective factors (aging, stress, thermal risk, BMS warnings, voltage imbalance, resistance, age, mileage, fast charging, interruptions/overcharge vs. maintenance, charging quality, thermal health, cooling health) plus Gaussian noise, calibrated to a ~10% positive rate. No single threshold determines the label. |

---

## Data Quality Notes
- **Missing values:** ~3–5% per column (except IDs and target), simulating real-world telemetry gaps.
- **Outliers:** ~0.6% of rows in select sensor columns (`cell_temperature_max`, `internal_resistance`, `pack_voltage`, `voltage_imbalance`, `average_charge_power_kw`) carry sensor-glitch-style extreme multipliers.
- **Chemistry differences:** LTO and LFP degrade slowest and carry the lowest thermal-runaway risk and failure rate; NMC, NCA, and LMO degrade faster and carry higher risk — reflected consistently across `capacity_loss_percent`, `thermal_runaway_risk`, and `battery_failure`.
- **Brand/fleet differences:** Fleet vehicles show substantially higher failure rates than private vehicles due to higher mileage, more fast charging, and more aggressive driving.
