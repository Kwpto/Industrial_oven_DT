import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "baking_oven_telemetry.csv"
MODEL_PATH = BASE_DIR / "oven_xgboost_model.json"
RUL_MODEL_PATH = BASE_DIR / "oven_rul_model.json"
SHAP_PATH = BASE_DIR / "shap_explainer.joblib"

# Model configuration
FEATURE_COLS = [
    "Zone1_Temp_C",
    "Zone2_Temp_C",
    "Zone3_Temp_C",
    "Humidity_Pct",
    "Belt_Speed_m_min",
    "Gas_Flow_m3_h",
    "Dough_Inlet_Moisture_Pct",
    "Product_Load_kg_min",
    "Exhaust_Damper_Pct",
    "Heat_Recovery_Eff_Pct",
]

# Physical constraints and thresholds
BELT_SPEED_MIN = 0.1
ANOMALY_THRESHOLD = 0.5
ZONE2_TARGET_TEMP = 210.0
MOISTURE_TARGET_PCT = 4.0

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_URL = f"http://127.0.0.1:{API_PORT}/predict"
