import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from xgboost import XGBClassifier, XGBRegressor

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Baking Oven Digital Twin",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Industrial dark theme CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; }
    .stMetric { background-color: #1e1e1e; border-radius: 8px; padding: 10px; }
    .stMetric label { color: #aaaaaa; font-size: 0.85rem; }
    .stMetric div { color: #ffffff; font-size: 1.4rem; font-weight: bold; }
    h1, h2, h3, h4 { color: #ffffff; }
    .stAlert { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load data and models
# ---------------------------------------------------------------------------
DATA_PATH = "baking_oven_telemetry.csv"
MODEL_PATH = "oven_xgboost_model.json"
RUL_MODEL_PATH = "oven_rul_model.json"
SHAP_PATH = "shap_explainer.joblib"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


@st.cache_resource
def load_models():
    clf = XGBClassifier()
    clf.load_model(MODEL_PATH)
    rul = XGBRegressor()
    rul.load_model(RUL_MODEL_PATH)
    shap_bundle = joblib.load(SHAP_PATH) if os.path.exists(SHAP_PATH) else None
    return clf, rul, shap_bundle


df = load_data()
clf, rul_model, shap_bundle = load_models()

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

# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------
def compute_physics(t1, t2, t3, belt_speed, gas_flow, dough_moisture, humidity, heat_recovery):
    t_res = (24.0 / belt_speed) * 60.0
    t_avg_k = ((t1 + t2 + t3) / 3.0) + 273.15
    k_rate = 1.2 * np.exp(-16000.0 / (8.314 * t_avg_k))
    moisture = (
        dough_moisture
        * np.exp(-k_rate * t_res)
        * (1 + 0.002 * (humidity - 40))
        * (1 - 0.003 * (heat_recovery - 70))
    )
    browning = 25.0 + 0.30 * (t3 - 140.0) * (t_res / 60.0)
    sec = (gas_flow * 38000.0) / (belt_speed * 60.0 * 2.5)
    return moisture, browning, sec


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("🔧 Control Room")
st.sidebar.markdown("### Timeline Scrubber")

if "timestamp_idx" not in st.session_state:
    st.session_state.timestamp_idx = 0


def _on_slider_change():
    st.session_state.timestamp_idx = st.session_state.ts_slider
    st.session_state.ts_num = st.session_state.ts_slider


def _on_num_change():
    st.session_state.timestamp_idx = int(st.session_state.ts_num)
    st.session_state.ts_slider = int(st.session_state.ts_num)


col_slider, col_number = st.sidebar.columns([3, 1])
with col_slider:
    st.slider(
        "Timestamp Index",
        min_value=0,
        max_value=len(df) - 1,
        value=st.session_state.timestamp_idx,
        step=1,
        key="ts_slider",
        on_change=_on_slider_change,
    )
with col_number:
    st.number_input(
        "Go to",
        min_value=0,
        max_value=len(df) - 1,
        value=st.session_state.timestamp_idx,
        step=1,
        key="ts_num",
        on_change=_on_num_change,
    )

timestamp_idx = st.session_state.timestamp_idx

row = df.iloc[timestamp_idx]
current_belt_speed = float(row["Belt_Speed_m_min"])

st.sidebar.markdown("### Belt Speed Override")
belt_speed_override = st.sidebar.slider(
    "Belt Speed (m/min)",
    min_value=8.0,
    max_value=16.0,
    value=current_belt_speed,
    step=0.1,
)

# ---------------------------------------------------------------------------
# Recompute KPIs with overridden belt speed
# ---------------------------------------------------------------------------
t1 = float(row["Zone1_Temp_C"])
t2 = float(row["Zone2_Temp_C"])
t3 = float(row["Zone3_Temp_C"])
humidity = float(row["Humidity_Pct"])
gas_flow = float(row["Gas_Flow_m3_h"])
dough_moisture = float(row["Dough_Inlet_Moisture_Pct"])
product_load = float(row["Product_Load_kg_min"])
exhaust_damper = float(row["Exhaust_Damper_Pct"])
heat_recovery = float(row["Heat_Recovery_Eff_Pct"])

moisture, browning, sec = compute_physics(
    t1, t2, t3, belt_speed_override, gas_flow, dough_moisture, humidity, heat_recovery
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏭 Continuous Baking Oven — Digital Twin")
st.caption(f"Live simulation timestamp: `{row['Timestamp']}`  |  Index: `{timestamp_idx}`")

# ---------------------------------------------------------------------------
# KPI header row
# ---------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(label="Zone 2 Temperature", value=f"{t2:.1f} °C")
with col2:
    st.metric(label="Product Moisture", value=f"{moisture:.2f}%")
with col3:
    st.metric(label="Browning Index", value=f"{browning:.1f} BI")
with col4:
    st.metric(label="SEC", value=f"{sec:.0f} kJ/kg")
with col5:
    st.metric(label="OEE", value=f"{row['OEE_Pct']:.1f}%")

# ---------------------------------------------------------------------------
# ML inference
# ---------------------------------------------------------------------------
feature_vector = pd.DataFrame(
    [[t1, t2, t3, humidity, belt_speed_override, gas_flow, dough_moisture, product_load, exhaust_damper, heat_recovery]],
    columns=FEATURE_COLS,
)
failure_proba = float(clf.predict_proba(feature_vector)[0, 1])
rul_hours = float(rul_model.predict(feature_vector)[0])

# ---------------------------------------------------------------------------
# Anomaly banner + RUL
# ---------------------------------------------------------------------------
st.divider()
banner_col, rul_col = st.columns([2, 1])

with banner_col:
    if failure_proba > 0.5:
        st.error(
            f"🚨 **ANOMALY DETECTED** — Failure probability: **{failure_proba * 100:.1f}%**\n\n"
            "Zone 2 burner misfire or quality defect predicted. Recommend immediate operator intervention."
        )
    else:
        st.success(
            f"✅ **OPERATIONAL** — Failure probability: **{failure_proba * 100:.1f}%**\n\n"
            "All zones within normal operating parameters."
        )

with rul_col:
    st.info(f"🔧 **Burner RUL**: `{rul_hours:.0f}` hours remaining")

# ---------------------------------------------------------------------------
# SHAP explanation
# ---------------------------------------------------------------------------
if shap_bundle is not None:
    st.subheader("🔍 Model Explainability — Top Feature Contributions")
    explainer = shap_bundle["explainer"]
    shap_values = explainer.shap_values(feature_vector)
    # Handle binary classification SHAP shape
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    sv = sv.reshape(-1)
    contrib = pd.DataFrame({"Feature": FEATURE_COLS, "Contribution": sv})
    contrib = contrib.reindex(contrib["Contribution"].abs().sort_values(ascending=False).index)
    contrib["Color"] = contrib["Contribution"].apply(lambda x: "#ff4b4b" if x > 0 else "#2ecc71")
    fig_shap = go.Figure(
        go.Bar(
            x=contrib["Contribution"],
            y=contrib["Feature"],
            orientation="h",
            marker_color=contrib["Color"],
        )
    )
    fig_shap.update_layout(
        title="SHAP-like feature impact on defect probability",
        xaxis_title="Impact on failure probability",
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        height=350,
    )
    st.plotly_chart(fig_shap, width="stretch")

# ---------------------------------------------------------------------------
# What-if simulator
# ---------------------------------------------------------------------------
st.subheader("🔮 What-If Simulator")
with st.expander("Open simulator"):
    sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)
    with sim_col1:
        sim_t2 = st.slider("Sim Zone2 Temp (°C)", 150.0, 230.0, t2, 0.5)
    with sim_col2:
        sim_belt = st.slider("Sim Belt Speed (m/min)", 8.0, 16.0, belt_speed_override, 0.1)
    with sim_col3:
        sim_humidity = st.slider("Sim Humidity (%)", 30.0, 50.0, humidity, 0.5)
    with sim_col4:
        sim_dough = st.slider("Sim Dough Moisture (%)", 35.0, 50.0, dough_moisture, 0.5)

    sim_vector = pd.DataFrame(
        [[t1, sim_t2, t3, sim_humidity, sim_belt, gas_flow, sim_dough, product_load, exhaust_damper, heat_recovery]],
        columns=FEATURE_COLS,
    )
    sim_proba = float(clf.predict_proba(sim_vector)[0, 1])
    sim_moisture, sim_browning, sim_sec = compute_physics(
        t1, sim_t2, t3, sim_belt, gas_flow, sim_dough, sim_humidity, heat_recovery
    )

    st.markdown(
        f"**Predicted failure probability:** `{sim_proba * 100:.1f}%` | "
        f"**Moisture:** `{sim_moisture:.2f}%` | "
        f"**Browning:** `{sim_browning:.1f} BI` | "
        f"**SEC:** `{sim_sec:.0f} kJ/kg`"
    )

# ---------------------------------------------------------------------------
# Alarm history
# ---------------------------------------------------------------------------
st.subheader("📋 Alarm History")
window_start = max(0, timestamp_idx - 299)
history_df = df.iloc[window_start : timestamp_idx + 1].copy()
history_df["Failure_Probability"] = clf.predict_proba(history_df[FEATURE_COLS])[:, 1]
history_df["Alarm"] = (history_df["Failure_Probability"] > 0.5).astype(int)
alarm_df = history_df[history_df["Alarm"] == 1][["Timestamp", "Zone2_Temp_C", "Failure_Probability"]].copy()
alarm_df["Failure_Probability"] = alarm_df["Failure_Probability"].apply(lambda x: f"{x * 100:.1f}%")

if alarm_df.empty:
    st.write("No alarms in the last 300 seconds.")
else:
    st.dataframe(alarm_df.reset_index(drop=True), width="stretch")
    st.download_button(
        label="Download alarm history CSV",
        data=alarm_df.to_csv(index=False),
        file_name="alarm_history.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Plotly charts
# ---------------------------------------------------------------------------
window_start = max(0, timestamp_idx - 99)
window_df = df.iloc[window_start : timestamp_idx + 1].copy()

# Chart A: 3-zone temperature trends
fig_temp = go.Figure()
fig_temp.add_trace(go.Scatter(x=window_df["Timestamp"], y=window_df["Zone1_Temp_C"], mode="lines", name="Zone 1", line=dict(color="#1f77b4")))
fig_temp.add_trace(go.Scatter(x=window_df["Timestamp"], y=window_df["Zone2_Temp_C"], mode="lines", name="Zone 2", line=dict(color="#ff7f0e")))
fig_temp.add_trace(go.Scatter(x=window_df["Timestamp"], y=window_df["Zone3_Temp_C"], mode="lines", name="Zone 3", line=dict(color="#2ca02c")))
fig_temp.update_layout(
    title="Real-Time 3-Zone Temperature Trends (Rolling 100s Window)",
    xaxis_title="Timestamp",
    yaxis_title="Temperature (°C)",
    hovermode="x unified",
    template="plotly_dark",
    height=450,
)

# Chart B: Moisture vs Browning
fig_quality = go.Figure()
fig_quality.add_trace(go.Scatter(x=window_df["Timestamp"], y=window_df["Moisture_Content_Pct"], mode="lines", name="Moisture (%)", line=dict(color="#17becf"), yaxis="y1"))
fig_quality.add_trace(go.Scatter(x=window_df["Timestamp"], y=window_df["Browning_Index_BI"], mode="lines", name="Browning Index", line=dict(color="#d62728"), yaxis="y2"))
fig_quality.update_layout(
    title="Moisture Evaporation vs Maillard Browning Progression",
    xaxis_title="Timestamp",
    yaxis=dict(title="Moisture (%)", side="left", color="#17becf"),
    yaxis2=dict(title="Browning Index (BI)", overlaying="y", side="right", color="#d62728"),
    hovermode="x unified",
    template="plotly_dark",
    height=450,
    legend=dict(x=0.01, y=0.99),
)

st.plotly_chart(fig_temp, width="stretch")
st.plotly_chart(fig_quality, width="stretch")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Industrial IoT Digital Twin | Physics: Arrhenius moisture + Maillard browning + heat transfer delay | "
    "ML: XGBoost quality classifier + SHAP + RUL regressor"
)
