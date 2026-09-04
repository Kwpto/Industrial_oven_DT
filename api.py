"""
FastAPI backend for the Industrial Oven Digital Twin.
Provides real-time inference endpoints for quality defects and burner RUL.
"""

import os
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from xgboost import XGBClassifier, XGBRegressor

app = FastAPI(title="Britannia Oven Digital Twin API", version="2.0.0")

DATA_PATH = "baking_oven_telemetry.csv"
MODEL_PATH = "oven_xgboost_model.json"
RUL_MODEL_PATH = "oven_rul_model.json"
SHAP_PATH = "shap_explainer.joblib"

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

# Load models at startup
clf = XGBClassifier()
clf.load_model(MODEL_PATH)

rul_model = XGBRegressor()
rul_model.load_model(RUL_MODEL_PATH)

shap_bundle = joblib.load(SHAP_PATH) if os.path.exists(SHAP_PATH) else None


class TelemetryPayload(BaseModel):
    Zone1_Temp_C: float
    Zone2_Temp_C: float
    Zone3_Temp_C: float
    Humidity_Pct: float
    Belt_Speed_m_min: float
    Gas_Flow_m3_h: float
    Dough_Inlet_Moisture_Pct: float
    Product_Load_kg_min: float
    Exhaust_Damper_Pct: float
    Heat_Recovery_Eff_Pct: float


class BatchPayload(BaseModel):
    readings: List[TelemetryPayload]


@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": True}


@app.post("/predict")
def predict(payload: TelemetryPayload):
    df = pd.DataFrame([payload.dict()], columns=FEATURE_COLS)
    failure_proba = float(clf.predict_proba(df)[0, 1])
    failure_flag = int(failure_proba > 0.5)
    rul = float(rul_model.predict(df)[0])

    response = {
        "failure_probability": round(failure_proba, 4),
        "failure_flag": failure_flag,
        "rul_burner_hours": round(rul, 2),
    }

    if shap_bundle is not None:
        explainer = shap_bundle["explainer"]
        sv = explainer.shap_values(df)
        if isinstance(sv, list):
            sv = sv[1]
        sv = sv.reshape(-1)
        response["shap_contributions"] = {
            feat: round(float(val), 6) for feat, val in zip(FEATURE_COLS, sv)
        }

    return response


@app.post("/predict/batch")
def predict_batch(payload: BatchPayload):
    rows = [r.dict() for r in payload.readings]
    df = pd.DataFrame(rows, columns=FEATURE_COLS)
    probas = clf.predict_proba(df)[:, 1]
    flags = (probas > 0.5).astype(int)
    ruls = rul_model.predict(df)

    return {
        "count": len(rows),
        "results": [
            {
                "failure_probability": round(float(p), 4),
                "failure_flag": int(f),
                "rul_burner_hours": round(float(r), 2),
            }
            for p, f, r in zip(probas, flags, ruls)
        ],
    }


@app.get("/telemetry/latest")
def latest_telemetry():
    df = pd.read_csv(DATA_PATH)
    latest = df.iloc[-1].to_dict()
    latest["Timestamp"] = str(latest["Timestamp"])
    return latest
