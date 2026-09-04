import numpy as np
import pandas as pd

# Set random seed for reproducible data
np.random.seed(42)
n_samples = 3000
timestamps = pd.date_range(start="2026-09-01 08:00:00", periods=n_samples, freq="1s")

# ---------------------------------------------------------------------------
# Operating parameters
# ---------------------------------------------------------------------------
belt_speed = np.random.normal(loc=12.0, scale=0.1, size=n_samples)  # m/min
humidity = np.random.normal(loc=40.0, scale=1.5, size=n_samples)    # %

# Multi-Zone Temperatures (°C) — realistic industrial variation
t1_base = np.random.normal(loc=160.0, scale=2.5, size=n_samples)
t2_base = np.random.normal(loc=210.0, scale=3.0, size=n_samples)
t3_base = np.random.normal(loc=180.0, scale=2.0, size=n_samples)

# ---------------------------------------------------------------------------
# Advanced process variables
# ---------------------------------------------------------------------------
dough_inlet_moisture = np.random.normal(loc=42.0, scale=2.0, size=n_samples)  # %
product_load = np.random.normal(loc=30.0, scale=1.0, size=n_samples)        # kg/min
exhaust_damper = np.clip(np.random.normal(loc=55.0, scale=5.0, size=n_samples), 20, 90)  # %
heat_recovery_eff = np.clip(np.random.normal(loc=70.0, scale=3.0, size=n_samples), 50, 95)  # %

# ---------------------------------------------------------------------------
# Injected fault scenarios (target ~8-12% defect rate)
# ---------------------------------------------------------------------------

# Fault 1: Zone 2 burner misfire (rows 1500-1800): 25 °C drop
t2_base[1500:1800] -= np.linspace(0, 25, 300)

# Fault 2: Belt speed drops 40% → over-baking (rows 500-700)
belt_speed[500:700] *= 0.60

# Fault 3: Humidity spikes + dough moisture spikes (rows 2000-2100)
humidity[2000:2100] += np.linspace(0, 18, 100)
dough_inlet_moisture[2000:2100] += np.linspace(0, 10, 100)

# Fault 4: Random extreme outliers (~50 rows)
outlier_idx = np.random.choice(n_samples, size=50, replace=False)
t2_base[outlier_idx] += np.random.choice([-1, 1], size=50) * np.random.uniform(15, 25, size=50)
belt_speed[outlier_idx] *= np.random.uniform(0.5, 0.7, size=50)

# Fault 5: Exhaust damper stuck closed (rows 2500-2600) → poor ventilation
exhaust_damper[2500:2600] = 20.0

# ---------------------------------------------------------------------------
# Heat transfer delay: Zone 3 is partly driven by lagged Zone 2 temperature
# ---------------------------------------------------------------------------
lag_seconds = 15
t3_effective = 0.75 * t3_base + 0.25 * np.roll(t2_base, lag_seconds)
t3_effective[:lag_seconds] = t3_base[:lag_seconds]

# ---------------------------------------------------------------------------
# Residence time inside 24m tunnel oven (seconds)
# ---------------------------------------------------------------------------
residence_time_sec = (24.0 / belt_speed) * 60.0

# ---------------------------------------------------------------------------
# Physics: Moisture Evaporation Kinetics (Arrhenius Rate Equation)
# ---------------------------------------------------------------------------
t_avg_k = ((t1_base + t2_base + t3_effective) / 3.0) + 273.15
R = 8.314
k_rate = 1.2 * np.exp(-16000.0 / (R * t_avg_k))

# Moisture depends on inlet moisture, residence time, humidity, and heat recovery
moisture_pct = (
    dough_inlet_moisture
    * np.exp(-k_rate * residence_time_sec)
    * (1 + 0.002 * (humidity - 40))
    * (1 - 0.003 * (heat_recovery_eff - 70))
    + np.random.normal(0, 0.08, n_samples)
)

# ---------------------------------------------------------------------------
# Physics: Maillard Browning Index
# ---------------------------------------------------------------------------
browning_index = (
    25.0
    + 0.30 * (t3_effective - 140) * (residence_time_sec / 60.0)
    + 0.05 * (exhaust_damper - 55)
    + np.random.normal(0, 0.4, n_samples)
)

# ---------------------------------------------------------------------------
# Fuel/Gas Consumption (m³/h)
# ---------------------------------------------------------------------------
gas_flow = (
    0.05 * (t1_base + t2_base * 1.5 + t3_effective) * (belt_speed / 12.0)
    * (1 + 0.005 * (product_load - 30))
    * (1 - 0.004 * (heat_recovery_eff - 70))
    + np.random.normal(0, 0.4, n_samples)
)

# ---------------------------------------------------------------------------
# OEE components
# ---------------------------------------------------------------------------
availability = np.clip(np.random.normal(loc=96.0, scale=1.5, size=n_samples), 85, 100)
performance = np.clip((belt_speed / 12.0) * 100, 85, 105)
quality = np.clip(100.0 - 100.0 * ((moisture_pct > 5.5) | (moisture_pct < 3.5) | (browning_index > 58.0) | (browning_index < 48.0)).astype(float), 0, 100)
oee_pct = (availability * performance * quality) / 10000.0

# ---------------------------------------------------------------------------
# Defect Flag (1 = Anomaly/Defect, 0 = Normal)
# Realistic thresholds calibrated to ~8-12% defect rate
# ---------------------------------------------------------------------------
defect_flag = (
    (moisture_pct > 5.5)           # too wet / under-baked
    | (moisture_pct < 3.5)         # too dry / over-baked
    | (browning_index > 58.0)      # burnt
    | (browning_index < 48.0)      # under-baked pale
    | (t2_base < 188.0)            # burner misfire
    | (t2_base > 228.0)            # dangerous over-temp
    | (oee_pct < 82.0)             # severe OEE drop
).astype(int)

# ---------------------------------------------------------------------------
# Remaining Useful Life (RUL) proxy for Zone 2 burner
# Cumulative thermal stress accelerates degradation during the misfire window
# ---------------------------------------------------------------------------
thermal_stress = np.zeros(n_samples)
for i in range(1, n_samples):
    stress = max(0, (t2_base[i] - 200) / 1000.0)
    # Misfire induces thermal shock / cycling stress
    if 1500 <= i < 1800:
        stress += 0.05
    thermal_stress[i] = thermal_stress[i - 1] + stress

rul_burner_hours = np.clip(2000 - thermal_stress * 10 + np.random.normal(0, 20, n_samples), 50, 2000)

# Create DataFrame
df = pd.DataFrame({
    "Timestamp": timestamps,
    "Zone1_Temp_C": np.round(t1_base, 2),
    "Zone2_Temp_C": np.round(t2_base, 2),
    "Zone3_Temp_C": np.round(t3_effective, 2),
    "Humidity_Pct": np.round(humidity, 2),
    "Belt_Speed_m_min": np.round(belt_speed, 2),
    "Gas_Flow_m3_h": np.round(gas_flow, 2),
    "Dough_Inlet_Moisture_Pct": np.round(dough_inlet_moisture, 2),
    "Product_Load_kg_min": np.round(product_load, 2),
    "Exhaust_Damper_Pct": np.round(exhaust_damper, 2),
    "Heat_Recovery_Eff_Pct": np.round(heat_recovery_eff, 2),
    "Moisture_Content_Pct": np.round(moisture_pct, 2),
    "Browning_Index_BI": np.round(browning_index, 2),
    "OEE_Pct": np.round(oee_pct, 2),
    "RUL_Burner_Hours": np.round(rul_burner_hours, 2),
    "Defect_Flag": defect_flag,
})

# Export to both CSV and Excel formats
df.to_csv("baking_oven_telemetry.csv", index=False)
df.to_excel("baking_oven_telemetry.xlsx", index=False)

print("SUCCESS: 'baking_oven_telemetry.xlsx' and 'baking_oven_telemetry.csv' generated in your folder!")
print(f"Rows: {len(df)} | Columns: {df.shape[1]}")
print(f"Defect rate: {df['Defect_Flag'].mean()*100:.2f}%")