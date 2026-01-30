import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta, timezone
import altair as alt
import requests
import io

# --------------------------------
# Helper funksiyalar
# -----------------------------
from functions import air_quality, get_weather, add_time_features

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(
    page_title="Uzbekistan Air Quality & Storm Prediction",
    layout="wide",
)
st.markdown("This app shows **air quality (PM2.5)** and **weather forecast** with simple storm warnings.")

# -----------------------------
# Auto-refresh (30 min)
# -----------------------------
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = datetime.now(timezone.utc)

now = datetime.now(timezone.utc)
if (now - st.session_state["last_refresh"]) > timedelta(minutes=30):
    st.session_state["last_refresh"] = now
    st.rerun()

# -----------------------------
# Model loader
# -----------------------------
@st.cache_data(ttl=600)
def load_model(path="best_model.pkl"):
    try:
        model = joblib.load(path)
        return model, None
    except Exception as e:
        return None, str(e)

# -----------------------------
# Forecast prep
# -----------------------------
def prepare_forecast_for_model(forecast_df, features):
    forecast_df = forecast_df.copy()
    forecast_df['date'] = pd.to_datetime(forecast_df['date'], utc=True, errors='coerce')
    forecast_feat = add_time_features(forecast_df, 'date')
    for col in ['value_lag1','value_lag24','value_roll6','value_roll24']:
        if col not in forecast_feat.columns:
            forecast_feat[col] = np.nan
    X_forecast = forecast_feat[[c for c in features if c in forecast_feat.columns]]
    return forecast_df, forecast_feat, X_forecast

# -----------------------------
# Disaster warning rules
# -----------------------------
def compute_disaster_warnings(df):
    warnings = []
    dfp = df.set_index('date').sort_index()

    if 'surface_pressure' in df.columns and len(dfp) >= 4:
        dfp['p_3h_diff'] = dfp['surface_pressure'] - dfp['surface_pressure'].shift(3)
        recent = dfp['p_3h_diff'].iloc[-1]
        if recent <= -6:
            warnings.append("⚠️ Pressure dropped ≥6 hPa in last 3h → possible storm front 🌪")

    if 'windspeed' in df.columns:
        max_w = df['windspeed'].max()
        if max_w >= 15:
            warnings.append("💨 Strong wind forecast (≥15 m/s) → storm risk")

    if 'precipitation' in df.columns and len(dfp) >= 24:
        tot24 = dfp['precipitation'].rolling(24, min_periods=1).sum().iloc[-1]
        if tot24 >= 20:
            warnings.append("🌧 Heavy rainfall (≥20 mm/24h) → flooding risk")

    if 'temperature_2m' in df.columns and len(dfp) >= 6:
        temp_diff = dfp['temperature_2m'].iloc[-1] - dfp['temperature_2m'].iloc[-6]
        if temp_diff <= -5:
            warnings.append(f"🌡 Sudden temp drop ({temp_diff:.1f}°C in 6h) → storm front signal")

    if 'temperature_2m' in df.columns:
        if df['temperature_2m'].max() > 40:
            warnings.append("🔥 Extreme heat (>40°C) → heatwave risk")

    if 'humidity' in df.columns and 'temperature_2m' in df.columns:
        last_hum = df['humidity'].iloc[-1]
        last_temp = df['temperature_2m'].iloc[-1]
        if last_hum > 85 and last_temp > 30:
            warnings.append("🥵 High humidity + heat → heat stress risk")

    if 'temperature_2m' in df.columns and 'precipitation' in df.columns:
        if df['temperature_2m'].min() < 0 and df['precipitation'].sum() > 0:
            warnings.append("❄️ Subzero + precipitation → icing/snowfall risk")

    if 'cloudcover' in df.columns and 'humidity' in df.columns:
        last_cc = df['cloudcover'].iloc[-1]
        last_h = df['humidity'].iloc[-1]
        if last_cc > 90 and last_h > 80:
            warnings.append("🌫 High humidity + cloudcover → fog risk")

    return warnings

def _certainty_label(score: float) -> str:
    if score >= 0.8:
        return "High"
    if score >= 0.6:
        return "Medium"
    if score >= 0.4:
        return "Low"
    return "Very Low"

def compute_certainty_scores(X_forecast: pd.DataFrame) -> pd.Series:
    if X_forecast is None or X_forecast.empty:
        return pd.Series(dtype=float)
    missing_ratio = X_forecast.isna().mean(axis=1)
    score = 1.0 - (missing_ratio * 0.9)
    score = score.clip(lower=0.0, upper=1.0)
    score = score.where(missing_ratio < 1.0, 0.0)
    return score

def compute_risk_alerts(forecast_df_prepared: pd.DataFrame, certainty_scores: pd.Series):
    alerts = []
    if forecast_df_prepared is None or forecast_df_prepared.empty:
        return alerts

    dfp = forecast_df_prepared.set_index('date').sort_index()
    if 'precipitation' in dfp.columns:
        rain_6h = dfp['precipitation'].rolling(6, min_periods=1).sum().max()
        rain_24h = dfp['precipitation'].rolling(24, min_periods=1).sum().max()
    else:
        rain_6h, rain_24h = np.nan, np.nan

    if 'temperature_2m' in dfp.columns:
        max_temp = dfp['temperature_2m'].max()
        min_temp = dfp['temperature_2m'].min()
    else:
        max_temp, min_temp = np.nan, np.nan

    if 'humidity' in dfp.columns:
        min_humidity = dfp['humidity'].min()
    else:
        min_humidity = np.nan

    if 'precipitation' in dfp.columns:
        total_72h = dfp['precipitation'].rolling(72, min_periods=1).sum().iloc[-1]
    else:
        total_72h = np.nan

    base_certainty = float(np.nanmean(certainty_scores)) if certainty_scores is not None and len(certainty_scores) else 0.0
    if np.isnan(base_certainty):
        base_certainty = 0.0

    landslide_level = "Low"
    landslide_reason = "Not enough signal for heavy-rainfall-triggered landslide risk."
    if not np.isnan(rain_24h) and not np.isnan(rain_6h):
        if rain_24h >= 40 or rain_6h >= 20:
            landslide_level = "High"
            landslide_reason = f"Very heavy rainfall (max 6h={rain_6h:.1f} mm, max 24h={rain_24h:.1f} mm)."
        elif rain_24h >= 25 or rain_6h >= 12:
            landslide_level = "Medium"
            landslide_reason = f"Heavy rainfall (max 6h={rain_6h:.1f} mm, max 24h={rain_24h:.1f} mm)."
        else:
            landslide_reason = f"Rainfall not extreme (max 6h={rain_6h:.1f} mm, max 24h={rain_24h:.1f} mm)."

    drought_level = "Low"
    drought_reason = "Forecast does not indicate dry + hot stress conditions."
    if not np.isnan(total_72h) and not np.isnan(max_temp):
        if total_72h < 1 and max_temp >= 35:
            drought_level = "High"
            drought_reason = f"Very low rain in next 72h (≈{total_72h:.1f} mm) + high temperature (max {max_temp:.1f}°C)."
        elif total_72h < 5 and max_temp >= 30:
            drought_level = "Medium"
            drought_reason = f"Low rain in next 72h (≈{total_72h:.1f} mm) + warm temperatures (max {max_temp:.1f}°C)."
        else:
            drought_reason = f"Rain/temperature not extreme (72h rain≈{total_72h:.1f} mm, max temp={max_temp:.1f}°C)."

    crop_level = "Low"
    crop_reason = "No major crop-stress trigger detected in forecast window."
    if not np.isnan(max_temp) and max_temp >= 40:
        crop_level = "High"
        crop_reason = f"Extreme heat (max {max_temp:.1f}°C) may cause heat stress and yield loss."
    elif not np.isnan(min_temp) and min_temp <= 0:
        crop_level = "Medium"
        crop_reason = f"Cold/frost risk (min {min_temp:.1f}°C) can damage sensitive crops."
    elif not np.isnan(rain_24h) and rain_24h >= 40:
        crop_level = "Medium"
        crop_reason = f"Excess water risk (max 24h rain {rain_24h:.1f} mm) can harm crops and fields."
    elif not np.isnan(max_temp) and not np.isnan(min_humidity) and max_temp >= 35 and min_humidity <= 25:
        crop_level = "Medium"
        crop_reason = f"Hot + dry air (max temp {max_temp:.1f}°C, min humidity {min_humidity:.0f}%) may stress crops."

    alerts.append({
        "hazard": "Landslide",
        "risk_level": landslide_level,
        "certainty_score": round(base_certainty, 2),
        "certainty_label": _certainty_label(base_certainty),
        "rationale": landslide_reason,
    })
    alerts.append({
        "hazard": "Drought",
        "risk_level": drought_level,
        "certainty_score": round(base_certainty, 2),
        "certainty_label": _certainty_label(base_certainty),
        "rationale": drought_reason,
    })
    alerts.append({
        "hazard": "Crop failure",
        "risk_level": crop_level,
        "certainty_score": round(base_certainty, 2),
        "certainty_label": _certainty_label(base_certainty),
        "rationale": crop_reason,
    })

    return alerts

@st.cache_data(ttl=3600)
def fetch_satellite_png(
    lat: float,
    lon: float,
    date_utc: datetime,
    km_radius: float = 25.0,
    layer: str = "MODIS_Terra_CorrectedReflectance_TrueColor",
    width: int = 800,
    height: int = 500,
):
    if lat is None or lon is None or np.isnan(lat) or np.isnan(lon):
        return None, "Missing coordinates"

    deg_lat = km_radius / 111.0
    deg_lon = km_radius / (111.0 * max(np.cos(np.deg2rad(lat)), 1e-6))

    min_lat = lat - deg_lat
    max_lat = lat + deg_lat
    min_lon = lon - deg_lon
    max_lon = lon + deg_lon

    time_str = date_utc.strftime("%Y-%m-%d")

    params = {
        "service": "WMS",
        "request": "GetMap",
        "version": "1.3.0",
        "layers": layer,
        "styles": "",
        "format": "image/png",
        "transparent": "false",
        "width": int(width),
        "height": int(height),
        "crs": "EPSG:4326",
        "bbox": f"{min_lat},{min_lon},{max_lat},{max_lon}",
        "time": time_str,
    }

    url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    try:
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code != 200 or not resp.content:
            return None, f"HTTP {resp.status_code}"
        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type.lower():
            preview = resp.text[:300] if hasattr(resp, "text") else ""
            return None, f"Unexpected response type: {content_type}. {preview}"
        return resp.content, None
    except Exception as e:
        return None, str(e)

# -----------------------------
# Features
# -----------------------------
FEATURES = [
    'temperature_2m','apparent_temperature','precipitation','rain','snowfall',
    'humidity','surface_pressure','year','month','day','hour','dayofweek',
    'is_weekend','hour_sin','hour_cos','dow_sin','dow_cos',
    'value_lag1','value_lag24','value_roll6','value_roll24'
]

SATELLITE_LAYERS = {
    "MODIS True Color (Terra)": {
        "layer": "MODIS_Terra_CorrectedReflectance_TrueColor",
        "caption": "NASA GIBS: MODIS True Color (approximate)",
    },
    "VIIRS True Color (SNPP)": {
        "layer": "VIIRS_SNPP_CorrectedReflectance_TrueColor",
        "caption": "NASA GIBS: VIIRS True Color (approximate)",
    },
    "AIRS CO Total Column (Day)": {
        "layer": "AIRS_CO_Total_Column_Day",
        "caption": "NASA GIBS: AIRS CO Total Column (Day)",
    },
}

SATELLITE_LOCATIONS = {
    "Tashkent": {"lat": 41.3111, "lon": 69.2797},
    "Samarkand": {"lat": 39.6542, "lon": 66.9597},
    "Bukhara": {"lat": 39.7681, "lon": 64.4556},
    "Fergana": {"lat": 40.3894, "lon": 71.7864},
    "Nukus": {"lat": 42.4600, "lon": 59.6166},
    "Custom": {"lat": 41.3111, "lon": 69.2797},
}

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Controls")
st.sidebar.text("Data: OpenAQ + OpenMeteo")
st.sidebar.text("Refresh: every 30 minutes")
st.sidebar.text("Forecast: 3 days")
st.sidebar.text("Model: best_model.pkl")

st.sidebar.subheader("Satellite imagery")
show_satellite = st.sidebar.checkbox("Show satellite imagery", value=True)

sat_layer_name = st.sidebar.selectbox(
    "Layer",
    options=list(SATELLITE_LAYERS.keys()),
    index=0,
)

sat_location_name = st.sidebar.selectbox(
    "Location",
    options=list(SATELLITE_LOCATIONS.keys()),
    index=0,
)

default_coords = SATELLITE_LOCATIONS.get(sat_location_name, SATELLITE_LOCATIONS["Tashkent"])
sat_lat = float(default_coords["lat"])
sat_lon = float(default_coords["lon"])
if sat_location_name == "Custom":
    sat_lat = float(st.sidebar.number_input("Latitude", value=sat_lat, format="%.6f"))
    sat_lon = float(st.sidebar.number_input("Longitude", value=sat_lon, format="%.6f"))

sat_date = st.sidebar.date_input("Date (UTC)", value=datetime.now(timezone.utc).date())
sat_km_radius = float(st.sidebar.slider("Radius (km)", min_value=5, max_value=150, value=25, step=5))

# -----------------------------
# Title
# -----------------------------
st.title("🇺🇿 Uzbekistan — Air Quality & Weather Forecast")

tab_dashboard, tab_alerts, tab_uncertainty = st.tabs([
    "Integrated Data Dashboard",
    "Risk Alerts (with Certainty)",
    "Uncertainty Signaling",
])

# -----------------------------
# Data
# -----------------------------
with tab_dashboard:
    st.header("1) Data")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Realtime Air Quality")
        try:
            aq_df = air_quality()
            if not aq_df.empty:
                row = aq_df.iloc[0]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Temperature (°C)", row.get("t", np.nan))
                    st.metric("Humidity (%)", row.get("h", np.nan))
                with col2:
                    st.metric("Pressure (hPa)", row.get("p", np.nan))
                    st.metric("CO (mg/m³)", row.get("co", np.nan))
                with col3:
                    st.metric("Wind Speed (m/s)", row.get("w", np.nan))
                    st.metric("SO₂ (µg/m³)", row.get("so2", np.nan))
            else:
                st.warning("No realtime data available.")
        except Exception as e:
            st.error(f"Realtime air quality error: {e}")

    with col_b:
        st.subheader("Weather Forecast (3 days)")
        try:
            today = datetime.now(timezone.utc).date()
            _, forecast_df = get_weather(date_from=today,
                                         date_till=(today + timedelta(days=3)),
                                         chunk_days=7)
            st.success("✅ Forecast fetched")
            st.dataframe(forecast_df.head(10))
        except Exception as e:
            st.error(f"Forecast fetch failed: {e}")
            forecast_df = pd.DataFrame()

    st.subheader("Satellite Imagery (Unified View)")
    sat_col1, sat_col2 = st.columns([2, 1])
    with sat_col1:
        if show_satellite:
            sat_layer_cfg = SATELLITE_LAYERS.get(sat_layer_name, SATELLITE_LAYERS["MODIS True Color (Terra)"])
            sat_dt_utc = datetime.combine(sat_date, datetime.min.time(), tzinfo=timezone.utc)
            sat_img, sat_err = fetch_satellite_png(
                lat=sat_lat,
                lon=sat_lon,
                date_utc=sat_dt_utc,
                km_radius=sat_km_radius,
                layer=sat_layer_cfg["layer"],
            )
            if sat_img is not None:
                st.image(sat_img, caption=sat_layer_cfg["caption"], use_container_width=True)
            else:
                st.info(f"Satellite imagery unavailable: {sat_err}")
        else:
            st.info("Satellite imagery is disabled in the sidebar.")
    with sat_col2:
        st.write(f"Location: {sat_location_name} (lat={sat_lat:.4f}, lon={sat_lon:.4f})")
        st.write(f"Layer: {sat_layer_name}")
        st.write("If the satellite panel fails, the app will continue with forecast + model outputs.")

# -----------------------------
# Prediction
# -----------------------------
    st.header("2) Prediction & Graphs")

    model, err = load_model("best_model.pkl")

    if model is None or forecast_df.empty:
        st.info("Model or forecast data missing — cannot run prediction.")
        forecast_df_prepared = pd.DataFrame()
        certainty_scores = pd.Series(dtype=float)
    else:
        forecast_df_prepared, forecast_feat, X_forecast = prepare_forecast_for_model(forecast_df, FEATURES)
        certainty_scores = compute_certainty_scores(X_forecast)
        try:
            preds = model.predict(X_forecast)
            forecast_df_prepared['prediction'] = preds
            st.success("✅ Predictions done")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            forecast_df_prepared = pd.DataFrame()
            certainty_scores = pd.Series(dtype=float)

        st.subheader("PM2.5 Forecast (next hours)")
        df_pm = forecast_df_prepared.dropna(subset=["prediction"])
        if not df_pm.empty:
            pm_chart = alt.Chart(df_pm).mark_line(color="red").encode(
                x="date:T",
                y=alt.Y("prediction:Q",
                        title="PM2.5 (µg/m³)",
                        scale=alt.Scale(zero=False)),
                tooltip=["date:T","prediction:Q"]
            ).properties(width=800, height=300)
            st.altair_chart(pm_chart, use_container_width=True)
        else:
            st.warning("No PM2.5 prediction data available.")

        st.subheader("Weather Features (Temperature, Humidity, Wind, Cloudcover)")
        weather_cols_main = ["temperature_2m","humidity","windspeed","cloudcover"]
        df_main = forecast_df_prepared.melt(id_vars=["date"], value_vars=weather_cols_main,
                                            var_name="feature", value_name="value")
        chart_main = alt.Chart(df_main).mark_line().encode(
            x="date:T", y="value:Q", color="feature:N",
            tooltip=["date:T","feature:N","value:Q"]
        ).properties(width=800, height=300)
        st.altair_chart(chart_main, use_container_width=True)

        st.subheader("Precipitation (Rainfall in mm)")
        df_prec = forecast_df_prepared.melt(id_vars=["date"], value_vars=["precipitation"],
                                            var_name="feature", value_name="value")
        chart_prec = alt.Chart(df_prec).mark_bar(color="blue").encode(
            x="date:T", y="value:Q",
            tooltip=["date:T","value:Q"]
        ).properties(width=800, height=300)
        st.altair_chart(chart_prec, use_container_width=True)

        st.subheader("Surface Pressure (hPa)")
        df_press = forecast_df_prepared.melt(id_vars=["date"], value_vars=["surface_pressure"],
                                             var_name="feature", value_name="value")

        min_val = df_press["value"].min()
        max_val = df_press["value"].max()

        chart_press = alt.Chart(df_press).mark_line(color="green").encode(
            x="date:T",
            y=alt.Y("value:Q",
                    title="Surface Pressure (hPa)",
                    scale=alt.Scale(domain=[min_val - 5, max_val + 5], zero=False)),
            tooltip=["date:T","value:Q"]
        ).properties(width=800, height=300)

        st.altair_chart(chart_press, use_container_width=True)

        st.subheader("⚠️ Warnings")
        warnings = compute_disaster_warnings(forecast_df_prepared)
        if warnings:
            for w in warnings:
                if "storm" in w.lower() or "heat" in w.lower():
                    st.error("🔴 " + w)
                elif "wind" in w.lower() or "rain" in w.lower():
                    st.warning("🟠 " + w)
                else:
                    st.info("🟢 " + w)
        else:
            st.success("🟢 Stable — no immediate disaster warnings.")

        st.subheader("Model Certainty (Data Quality Proxy)")
        avg_certainty = float(np.nanmean(certainty_scores)) if len(certainty_scores) else 0.0
        if np.isnan(avg_certainty):
            avg_certainty = 0.0
        st.progress(avg_certainty)
        st.write(f"Certainty Score: {avg_certainty:.2f} ({_certainty_label(avg_certainty)})")

with tab_alerts:
    st.header("Risk Alerts")
    if 'forecast_df_prepared' in locals() and not forecast_df_prepared.empty:
        alerts = compute_risk_alerts(forecast_df_prepared, certainty_scores if 'certainty_scores' in locals() else pd.Series(dtype=float))
        for a in alerts:
            title = f"{a['hazard']} — {a['risk_level']} risk"
            body = f"Certainty: {a['certainty_score']} ({a['certainty_label']})\n\nReason: {a['rationale']}"
            if a["risk_level"] == "High":
                st.error(title + "\n\n" + body)
            elif a["risk_level"] == "Medium":
                st.warning(title + "\n\n" + body)
            else:
                st.info(title + "\n\n" + body)
    else:
        st.info("Risk alerts require forecast + prediction pipeline to run.")

with tab_uncertainty:
    st.header("Uncertainty Signaling")
    if 'forecast_df_prepared' not in locals() or forecast_df is None or forecast_df.empty:
        st.error("The system does not have enough data right now (forecast missing). Output is not reliable.")
    elif 'certainty_scores' not in locals() or len(certainty_scores) == 0:
        st.error("The system cannot compute a certainty score (model inputs not available). Treat results as uncertain.")
    else:
        avg_certainty = float(np.nanmean(certainty_scores))
        low_fraction = float((certainty_scores < 0.5).mean())
        if np.isnan(avg_certainty):
            avg_certainty = 0.0

        if avg_certainty < 0.4 or low_fraction > 0.5:
            st.error("Model does not know (high uncertainty). Do not treat predictions as accurate.")
        elif avg_certainty < 0.6:
            st.warning("Uncertainty is elevated. Treat predictions as approximate.")
        else:
            st.success("Uncertainty is low for current inputs.")

        st.write(f"Average certainty: {avg_certainty:.2f} ({_certainty_label(avg_certainty)})")
        st.write(f"Fraction of low-certainty timesteps (<0.5): {low_fraction:.2f}")
        st.dataframe(pd.DataFrame({
            "date": forecast_df_prepared["date"] if 'forecast_df_prepared' in locals() and 'date' in forecast_df_prepared.columns else [],
            "certainty_score": certainty_scores.values if len(certainty_scores) else [],
        }))

# -----------------------------
# Health advice
# -----------------------------
with tab_dashboard:
    st.header("3) Health Advice (PM2.5)")
    if 'forecast_df_prepared' in locals() and 'prediction' in forecast_df_prepared.columns and not forecast_df_prepared.empty:
        max_pred = forecast_df_prepared['prediction'].max()
        if max_pred > 150:
            st.error("🔴 Very Unhealthy — 😷 Avoid outdoors, wear N95.")
        elif max_pred > 100:
            st.warning("🟠 Unhealthy — 😷 Wear a mask, avoid long outdoor activity.")
        elif max_pred > 50:
            st.info("🟡 Moderate — Sensitive groups should limit outdoor exposure.")
        else:
            st.success("🟢 Safe — Air quality is good.")
    else:
        st.info("No prediction to assess health advice.")

# -----------------------------
# Glossary
# -----------------------------
with tab_dashboard:
    st.header("4) Glossary (Key Terms)")



    # CSS styling light/dark mode uchun
    st.markdown(
        """
        <style>
        /* Text rangini light va dark mode uchun */
        .glossary-item {
            margin-bottom: 10px;
            font-size: 16px;
        }
        [data-theme="light"] .glossary-item {
            color: black;
        }
        [data-theme="dark"] .glossary-item {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Glossary items
    glossary_items = {
        "🌫 PM2.5": "Tiny dust particles that can harm lungs.",
        "💧 Humidity (%)": "Amount of water vapor in the air.",
        "📉 Pressure (hPa)": "Air pressure; sudden drop → storm risk.",
        "🌬 Wind speed (m/s)": "How strong the wind blows.",
        "🌧 Rainfall (mm)": "Volume of rain; clears pollution but may flood.",
        "☁️ Cloud cover (%)": "How much of the sky is covered by clouds."
    }

    # Chiqarish
    for key, value in glossary_items.items():
        st.markdown(f"<div class='glossary-item'><b>{key}</b>: {value}</div>", unsafe_allow_html=True)


with tab_dashboard:
    st.markdown("---")
    st.caption("Built for Uzbekistan (Central Asia) — PM2.5 + weather forecasting & storm warnings.")
