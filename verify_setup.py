import os
import pandas as pd

print("=" * 40)
print("  DIGITAL TWIN PRE-FLIGHT READINESS CHECK")
print("=" * 40)

# 1. Check Libraries
try:
    import numpy
    import plotly
    import sklearn
    import streamlit
    import xgboost
    print("✅ Libraries: All required packages installed.")
except ImportError as e:
    print(f"❌ Libraries Missing: {e}")

# 2. Check Data File
if os.path.exists("baking_oven_telemetry.csv"):
    df = pd.read_csv("baking_oven_telemetry.csv")
    if len(df) == 3000 and df.shape[1] == 10:
        print(f"✅ Telemetry Data: Verified ({len(df)} rows, {df.shape[1]} columns).")
    else:
        print("⚠️ Telemetry Data: File found, but row/column count differs.")
else:
    print("❌ Telemetry Data: 'baking_oven_telemetry.csv' not found.")

# 3. Train & Verify XGBoost Model Artifact
try:
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split

    X = df[["Zone1_Temp_C", "Zone2_Temp_C", "Zone3_Temp_C", "Humidity_Pct", "Belt_Speed_m_min", "Gas_Flow_m3_h"]]
    y = df["Defect_Flag"]
    
    model = XGBClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X, y)
    model.save_model("oven_xgboost_model.json")
    
    if os.path.exists("oven_xgboost_model.json"):
        print("✅ Machine Learning: XGBoost model trained & saved to 'oven_xgboost_model.json'.")
except Exception as e:
    print(f"❌ Machine Learning Error: {e}")

print("=" * 40)
print("READY TO BUILD UI: Execute 'streamlit run app.py' when ready.")
print("=" * 40)