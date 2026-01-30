import requests
import pandas as pd
from datetime import datetime, timedelta
import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests
import time
import numpy as np

def air_quality():
    

    # Your token
    TOKEN = "a9ee66d6d0f4f1574968e2a90e5ddd462d1ded70"

    # Example 1: Get current feed (nearest station to your IP or given city/coords)
    url = f"https://api.waqi.info/feed/here/?token={TOKEN}"
    resp = requests.get(url, timeout=30)
    data = resp.json()

 
    # Example 2: Get data for a specific city (e.g. Tashkent)
    city = "tashkent"
    url_city = f"https://api.waqi.info/feed/{city}/?token={TOKEN}"
    resp_city = requests.get(url_city, timeout=30)
    data_city = resp_city.json()

    # Convert IAQI (individual pollutant readings) into a DataFrame
    iaqi = data_city["data"]["iaqi"]
    df = pd.DataFrame([{k: v["v"] for k, v in iaqi.items()}])
    return df
from datetime import datetime, timedelta
import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests
import time

# --- 1. Shaharlar ro'yxati ---
city_dict = {
    "Tashkent": (41.3111, 69.2797),
    #"Samarkand": (39.6542, 66.9597)
}

# --- 2. OpenMeteo API chaqiruvchi funksiya ---
def get_weather_openmeteo(lat, lon, start_date, end_date, mode="archive"):
    cache_session = requests_cache.CachedSession('.cache', expire_after=0)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    if mode == "archive":
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": [
                "temperature_2m",
                "apparent_temperature",
                "precipitation",
                "rain",
                "snowfall",
                "weather_code",
                "relative_humidity_2m",
                "windspeed_10m",
                "winddirection_10m",
                "surface_pressure",
                "cloudcover"
            ]
        }
    elif mode == "forecast":
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": datetime.utcnow().date(),
            "end_date": datetime.utcnow().date() + timedelta(days=7),
            "hourly": [
                "temperature_2m",
                "apparent_temperature",
                "precipitation",
                "rain",
                "snowfall",
                "weather_code",
                "relative_humidity_2m",
                "windspeed_10m",
                "winddirection_10m",
                "surface_pressure",
                "cloudcover"
            ],
            "timezone": "auto"
        }
    else:
        raise ValueError("Mode must be 'archive' or 'forecast'")

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "apparent_temperature": hourly.Variables(1).ValuesAsNumpy(),
        "precipitation": hourly.Variables(2).ValuesAsNumpy(),
        "rain": hourly.Variables(3).ValuesAsNumpy(),
        "snowfall": hourly.Variables(4).ValuesAsNumpy(),
        "weather_code": hourly.Variables(5).ValuesAsNumpy(),
        "humidity": hourly.Variables(6).ValuesAsNumpy(),
        "windspeed": hourly.Variables(7).ValuesAsNumpy(),
        "winddirection": hourly.Variables(8).ValuesAsNumpy(),
        "surface_pressure": hourly.Variables(9).ValuesAsNumpy(),
        "cloudcover": hourly.Variables(10).ValuesAsNumpy(),
    }
    return pd.DataFrame(data)
# 2) Feature engineering
def add_time_features(df, date_col='date'):
    out = df.copy()
    dt = out[date_col].dt.tz_convert('Asia/Tashkent') if out[date_col].dt.tz is not None else out[date_col]

    # Asosiy vaqt xususiyatlari
    out['year']       = dt.dt.year
    out['month']      = dt.dt.month
    out['day']        = dt.dt.day
    out['hour']       = dt.dt.hour
    out['dayofweek']  = dt.dt.dayofweek
    out['is_weekend'] = (out['dayofweek'] >= 5).astype(int)

    # Siklik kodlash
    out['hour_sin'] = np.sin(2*np.pi*out['hour']/24)
    out['hour_cos'] = np.cos(2*np.pi*out['hour']/24)
    out['dow_sin']  = np.sin(2*np.pi*out['dayofweek']/7)
    out['dow_cos']  = np.cos(2*np.pi*out['dayofweek']/7)

    # Lag/rolling
    out = out.set_index('date').copy()
    if 'value' in out.columns:
        out['value_lag1']  = out['value'].shift(1)
        out['value_lag24'] = out['value'].shift(24)
        out['value_roll6'] = out['value'].shift(1).rolling(6, min_periods=1).mean()
        out['value_roll24']= out['value'].shift(1).rolling(24, min_periods=1).mean()
    out = out.reset_index()
    return out

# --- 3. Umumiy get_weather() funksiyasi (bo‘lib olish) ---
def get_weather(date_from=None, date_till=None, chunk_days=180):
    if date_from is None:
        date_from = (datetime.utcnow().date() - timedelta(days=600))
    if date_till is None:
        date_till = datetime.utcnow().date()

    df_archive = pd.DataFrame()

    for city, (lat, lon) in city_dict.items():
        start = date_from
        while start < date_till:
            end = min(start + timedelta(days=chunk_days), date_till)

            # API chaqirish
            #df_city = get_weather_openmeteo(lat, lon, start, end, mode="archive")
            #df_city["city"] = city
            #df_archive = pd.concat([df_archive, df_city])

            print(f"✅ Got archive data: {city} {start} → {end}")
            start = end + timedelta(days=1)
            time.sleep(1)

    # Prognoz (forecast)
    df_forecast = pd.DataFrame()
    for city, (lat, lon) in city_dict.items():
        df_city = get_weather_openmeteo(lat, lon, date_from, date_till, mode="forecast")
        df_city["city"] = city
        df_forecast = pd.concat([df_forecast, df_city])
        print(f"✅ Got forecast data: {city}")
        time.sleep(1)

    return df_archive.reset_index(drop=True), df_forecast.reset_index(drop=True)
