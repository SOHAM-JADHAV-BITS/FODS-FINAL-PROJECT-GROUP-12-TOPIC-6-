import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler

#  CONFIGURATION SECTION
# (I usually keep my paths at the top so I don’t have to dig for them later.)
BASE_PATH = r"C:\Users\SOHAM\OneDrive\Desktop\venv"

N_STEPS_IN = 48                    # number of past rows the model looks at
FORECAST_HORIZONS = [12, 48, 72]   # hours ahead to forecast

# weights for the ensemble of models (XGB, RF, CAT)
ENSEMBLE_WEIGHTS = [0.34, 0.33, 0.33]

# pollutant and temporal features (I left this flat for readability)
POLLUTANTS = ['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3', 'no', 'nh3']
TEMPORAL   = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
MODEL_FEATURES = POLLUTANTS + TEMPORAL

# -------------------------------------------------------------------------
# AQI COLOR PALETTE (copy-paste ready)
#symbols are copy pasted from the web
AQI_COLORS = {
    "Good": "#00C853",        # 🟢
    "Moderate": "#FFD600",    # 🟡
    "Poor": "#FF9100",        # 🟠
    "Very Poor": "#D50000",   # 🔴
    "Hazardous": "#AA00FF"    # 🟣
}

def aqi_color(aqi: float) -> str:
    """Return emoji + label depending on AQI range."""
    if aqi <= 50:   return "🟢 Good"
    if aqi <= 100:  return "🟡 Moderate"
    if aqi <= 200:  return "🟠 Poor"
    if aqi <= 300:  return "🔴 Very Poor"
    return "🟣 Hazardous"

# HELPERS

def load_joblib(fname: str):
    """Quick wrapper to load .pkl files safely from the base path."""
    full_path = os.path.join(BASE_PATH, fname)
    return joblib.load(full_path)

def read_csv(fname: str):
    """Load CSV without worrying about memory optimization (classic oversight)."""
    full_path = os.path.join(BASE_PATH, fname)
    return pd.read_csv(full_path)

# Streamlit page setup
st.set_page_config(page_title="AQI Forecast Dashboard", layout="wide")


#  LOAD MODELS & DATA
try:
    xgb_model = load_joblib("xgb_model.pkl")
    rf_model  = load_joblib("rf_model.pkl")
    cat_model = load_joblib("cat_model.pkl")
except Exception as e:
    st.error(f" Could not load one or more models: {e}")
    st.stop()

try:
    df = read_csv("processed_data.csv")
except FileNotFoundError:
    st.error(" Could not find 'processed_data.csv' in the base path.")
    st.stop()

# Check columns
required_cols = set(MODEL_FEATURES + ["Net_AQI", "date"])
missing = required_cols - set(df.columns)
if missing:
    st.error(f" Missing columns: {sorted(list(missing))}")
    st.stop()

# Parse and clean dates
df["_parsed_date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["_parsed_date"]).reset_index(drop=True)

if len(df) < N_STEPS_IN:
    st.error(f" Need at least {N_STEPS_IN} rows; found only {len(df)}.")
    st.stop()

raw_date_str = str(df["date"].iloc[-1])
base_time = df["_parsed_date"].iloc[-1]

#  SCALER SETUP
# -------------------------------------------------------------------------
try:
    scaler = load_joblib("scaler.pkl")
except Exception:
    scaler = MinMaxScaler()
    scaler.fit(df[MODEL_FEATURES])
    try:
        joblib.dump(scaler, os.path.join(BASE_PATH, "scaler.pkl"))
    except Exception:
        # just ignore; not a critical failure
        pass


# PREPARE INPUT WINDOW (last 48 rows)
window_df = df.tail(N_STEPS_IN)
window = window_df[MODEL_FEATURES].values
window_scaled = scaler.transform(window)
window_flat = window_scaled.flatten().reshape(1, -1)


# ENSEMBLE PREDICTION
try:
    y_pred_xgb = xgb_model.predict(window_flat)
    y_pred_rf  = rf_model.predict(window_flat)
    y_pred_cat = cat_model.predict(window_flat)
except Exception as e:
    st.error(f" Model prediction error: {e}")
    st.stop()

w1, w2, w3 = ENSEMBLE_WEIGHTS
try:
    y_pred = (w1 * y_pred_xgb + w2 * y_pred_rf + w3 * y_pred_cat)[0]
except Exception:
    st.error(" Prediction shapes don't match expected structure.")
    st.stop()

if len(y_pred) < 3:
    st.error(" Models did not output 3 horizon predictions (expected 12h, 48h, 72h).")
    st.stop()

aqi_12, aqi_48, aqi_72 = map(float, y_pred[:3])

# DASHBOARD UI
header_left, header_right = st.columns([3, 1])
with header_left:
    st.markdown("AQI Forecast Dashboard**")
with header_right:
    st.markdown(f"{raw_date_str}")

st.markdown("--")

# Sidebar — latest pollutant levels
latest_row = df.iloc[-1]
st.sidebar.markdown(" Current Pollutant Levels")
for pollutant in POLLUTANTS:
    try:
        val = float(latest_row[pollutant])
        st.sidebar.write(f"**{pollutant.upper()}**: {val}")
    except Exception:
        st.sidebar.write(f"**{pollutant.upper()}**: -")


# FORECAST METRICS

def fmt_label(dt):
    return dt.strftime("%d-%m-%Y %I:%M %p")

t12 = base_time + timedelta(hours=12)
t48 = base_time + timedelta(hours=48)
t72 = base_time + timedelta(hours=72)

st.markdown(" Forecasted AQI Levels")
col1, col2, col3 = st.columns(3)
col1.metric(f"12h  {fmt_label(t12)}", f"{aqi_12}", aqi_color(aqi_12))
col2.metric(f"48h {fmt_label(t48)}", f"{aqi_48}", aqi_color(aqi_48))
col3.metric(f"72h  {fmt_label(t72)}", f"{aqi_72}", aqi_color(aqi_72))

st.markdown("-")

#  HISTORICAL CHART

st.markdown(" Last 48 Rows (AQI) + Forecast Points")

last48 = df.tail(N_STEPS_IN)[["_parsed_date", "Net_AQI"]].dropna().copy()
last48 = last48.set_index("_parsed_date").sort_index()

future_points = pd.Series(
    [aqi_12, aqi_48, aqi_72],
    index=[t12, t48, t72],
    name="Net_AQI"
)

series_to_plot = pd.concat([last48["Net_AQI"], future_points])
st.line_chart(series_to_plot.to_frame(name="AQI"))


# FOOTNOTE
# -------------------------------------------------------------------------
st.markdown("---")
st.markdown("Forecasts use the last 48 rows from the CSV as input to predict 12h, 48h, and 72h ahead.*")
