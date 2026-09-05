import os
import pandas as pd
from config import DATA_PATH, MODEL_PATH, RUL_MODEL_PATH, SHAP_PATH, FEATURE_COLS

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
    import joblib
    import shap
    import fastapi
    import uvicorn
    import requests
    import openpyxl
    print("✅ Libraries: All required packages installed.")
except ImportError as e:
    print(f"❌ Libraries Missing: {e}")

# 2. Check Data File
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    if len(df) == 3000 and df.shape[1] == 16:
        print(f"✅ Telemetry Data: Verified ({len(df)} rows, {df.shape[1]} columns).")
    else:
        print(f"⚠️ Telemetry Data: File found, but row/column count differs (Rows: {len(df)}, Cols: {df.shape[1]}).")
else:
    print(f"❌ Telemetry Data: '{DATA_PATH}' not found.")
    df = None

# 3. Verify XGBoost Model Artifacts
if df is not None:
    try:
        from xgboost import XGBClassifier, XGBRegressor

        if os.path.exists(MODEL_PATH):
            X = df[FEATURE_COLS]
            
            clf = XGBClassifier()
            clf.load_model(MODEL_PATH)
            
            test_pred = clf.predict(X[:5])
            print(f"✅ Quality Classifier: Loaded successfully (sample predictions: {test_pred})")
        else:
            print(f"❌ Quality Classifier: '{MODEL_PATH}' not found.")
            
        if os.path.exists(RUL_MODEL_PATH):
            rul = XGBRegressor()
            rul.load_model(RUL_MODEL_PATH)
            print(f"✅ RUL Regressor: Loaded successfully.")
        else:
            print(f"❌ RUL Regressor: '{RUL_MODEL_PATH}' not found.")
            
    except Exception as e:
        print(f"❌ Machine Learning Error: {e}")

print("=" * 40)
print("READY TO BUILD UI: Execute 'streamlit run app.py' when ready.")
print("=" * 40)