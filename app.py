import os
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Nigeria Weather Predictor", page_icon="🌦️")

# State capitals (latitude, longitude)
STATES = {
    "Abia": (5.53, 7.49), "Adamawa": (9.20, 12.48), "Akwa Ibom": (5.05, 7.93),
    "Anambra": (6.21, 7.07), "Bauchi": (10.31, 9.84), "Bayelsa": (4.93, 6.26),
    "Benue": (7.73, 8.52), "Borno": (11.85, 13.16), "Cross River": (4.95, 8.33),
    "Delta": (6.20, 6.73), "Ebonyi": (6.32, 8.11), "Edo": (6.34, 5.63),
    "Ekiti": (7.62, 5.22), "Enugu": (6.46, 7.55), "FCT (Abuja)": (9.08, 7.40),
    "Gombe": (10.29, 11.17), "Imo": (5.48, 7.03), "Jigawa": (11.76, 9.34),
    "Kaduna": (10.52, 7.44), "Kano": (12.00, 8.52), "Katsina": (12.99, 7.60),
    "Kebbi": (12.45, 4.20), "Kogi": (7.80, 6.74), "Kwara": (8.50, 4.55),
    "Lagos": (6.60, 3.35), "Nasarawa": (8.49, 8.52), "Niger": (9.61, 6.56),
    "Ogun": (7.16, 3.35), "Ondo": (7.25, 5.19), "Osun": (7.77, 4.56),
    "Oyo": (7.38, 3.95), "Plateau": (9.93, 8.89), "Rivers": (4.82, 7.03),
    "Sokoto": (13.06, 5.24), "Taraba": (8.89, 11.36), "Yobe": (11.75, 11.96),
    "Zamfara": (12.17, 6.66),
}

START_DATE = "2021-01-01"
END_DATE = "2025-12-31"
TIMES = {"Morning": 6, "Afternoon": 15, "Evening": 18}
HOUR_TO_TOD = {v: k for k, v in TIMES.items()}
CACHE_FILE = "nigeria_weather_2021_2025.csv"
API_URL = "https://archive-api.open-meteo.com/v1/archive"

TARGETS = {
    "temperature_C": "Temperature (°C)",
    "humidity_percent": "Humidity (%)",
    "pressure_hPa": "Pressure (hPa)",
    "wind_speed_mps": "Wind speed (m/s)",
}


def download_state(name, lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m",
        "wind_speed_unit": "ms",
        "timezone": "Africa/Lagos",
    }
    response = None
    for attempt in range(5):
        response = requests.get(API_URL, params=params, timeout=120)
        if response.status_code == 200:
            break
        time.sleep(20 * (attempt + 1))
    if response is None or response.status_code != 200:
        raise RuntimeError(f"Could not download {name}: {response.text[:200] if response is not None else ''}")

    h = response.json()["hourly"]
    d = pd.DataFrame({
        "time": pd.to_datetime(h["time"]),
        "temperature_C": h["temperature_2m"],
        "humidity_percent": h["relative_humidity_2m"],
        "pressure_hPa": h["pressure_msl"],
        "wind_speed_mps": h["wind_speed_10m"],
    })
    d = d[d["time"].dt.hour.isin(TIMES.values())].dropna()
    d["state"] = name
    d["latitude"] = lat
    d["longitude"] = lon
    return d


def download_all():
    bar = st.progress(0.0, text="Downloading 5 years of weather data (first run only)...")
    frames = []
    for i, (name, (lat, lon)) in enumerate(STATES.items()):
        frames.append(download_state(name, lat, lon))
        bar.progress((i + 1) / len(STATES), text=f"Downloaded {name} ({i + 1}/{len(STATES)})")
        time.sleep(3)
    pd.concat(frames, ignore_index=True).to_csv(CACHE_FILE, index=False)
    bar.empty()


@st.cache_data
def read_data():
    df = pd.read_csv(CACHE_FILE, parse_dates=["time"])
    df["date"] = df["time"].dt.normalize()
    df["time_of_day"] = df["time"].dt.hour.map(HOUR_TO_TOD)
    return df


def make_features(frame):
    doy = frame["date"].dt.dayofyear
    return pd.DataFrame({
        "latitude": frame["latitude"].values,
        "longitude": frame["longitude"].values,
        "hour": frame["time_of_day"].map(TIMES).values,
        "doy_sin": np.sin(2 * np.pi * doy / 365.25).values,
        "doy_cos": np.cos(2 * np.pi * doy / 365.25).values,
    })


def new_model():
    return RandomForestRegressor(n_estimators=80, min_samples_leaf=10, n_jobs=-1, random_state=42)


@st.cache_resource(show_spinner="Training the model...")
def train_model(_df):
    X, y = make_features(_df), _df[list(TARGETS)]
    train = (_df["date"].dt.year <= 2024).values
    test = ~train

    check = new_model().fit(X[train], y[train])
    pred = check.predict(X[test])
    rows = []
    for i, col in enumerate(TARGETS):
        rows.append({
            "What is predicted": TARGETS[col],
            "R2 score": round(r2_score(y[test][col], pred[:, i]), 3),
            "Average error": round(mean_absolute_error(y[test][col], pred[:, i]), 2),
        })

    final = new_model().fit(X, y)
    return final, pd.DataFrame(rows)


def predict_day(model, state, day):
    lat, lon = STATES[state]
    frame = pd.DataFrame({
        "date": [pd.Timestamp(day)] * 3,
        "time_of_day": list(TIMES),
        "latitude": lat,
        "longitude": lon,
    })
    out = pd.DataFrame(model.predict(make_features(frame)), columns=list(TARGETS))
    out.insert(0, "Time of day", list(TIMES))
    return out


# ---------------------------- APP ----------------------------
st.title("🌦️ Nigeria Weather Predictor")
st.write(
    "A machine learning app that predicts the typical temperature, humidity, pressure and "
    "wind for any Nigerian state on any date, learned from 5 years of real weather data (2021 to 2025)."
)

if not os.path.exists(CACHE_FILE):
    try:
        download_all()
    except Exception as e:
        st.error("Could not download the weather data. Check your internet connection and refresh the page.")
        st.caption(str(e))
        st.stop()

df = read_data()
model, metrics = train_model(df)

tab1, tab2, tab3 = st.tabs(["Predict", "Model accuracy", "Data"])

with tab1:
    state = st.selectbox("State", list(STATES), index=list(STATES).index("Kaduna"))
    day = st.date_input("Date", value=date.today() + timedelta(days=1),
                        min_value=date(2026, 1, 1), max_value=date(2035, 12, 31))
    time_of_day = st.radio("Time of day", list(TIMES), index=1, horizontal=True)

    if st.button("Predict weather", type="primary"):
        result = predict_day(model, state, day)
        st.subheader(f"{state}: {pd.Timestamp(day).day_name()}, {day.strftime('%d %B %Y')}")
        now = result[result["Time of day"] == time_of_day].iloc[0]
        c1, c2 = st.columns(2)
        c1.metric("Temperature", f"{now['temperature_C']:.1f} °C")
        c2.metric("Humidity", f"{now['humidity_percent']:.0f} %")
        c3, c4 = st.columns(2)
        c3.metric("Pressure", f"{now['pressure_hPa']:.1f} hPa")
        c4.metric("Wind speed", f"{now['wind_speed_mps']:.1f} m/s")

        st.write("The whole day:")
        table = result.rename(columns=TARGETS).round(1)
        st.dataframe(table, hide_index=True)
        st.caption(
            "This is the typical weather for this state at this time of year, learned from "
            "past years. It is not an exact forecast for the day."
        )

with tab2:
    st.write(
        "The model learned from 2021 to 2024, then was tested on 2025, a year it had never seen. "
        "A higher R2 score (max 1.0) and a lower average error mean a better model. "
        "The average error is in the same unit as the thing predicted."
    )
    st.dataframe(metrics, hide_index=True)
    st.write("Model used: Random Forest Regression.")

with tab3:
    st.write(f"{len(df):,} readings from {df['date'].min().date()} to {df['date'].max().date()} (source: Open-Meteo).")
    chosen = st.selectbox("Show monthly average temperature for", list(STATES), key="chart_state")
    sub = df[df["state"] == chosen]
    monthly = sub.groupby([sub["time"].dt.month, "time_of_day"])["temperature_C"].mean().unstack()
    st.line_chart(monthly)
    st.caption("Horizontal axis: month of the year (1 = January).")