# -------------------- SUPPRESS WARNINGS --------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger('absl').setLevel(logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')


# -------------------- IMPORTS --------------------
from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
import joblib
import requests
from datetime import datetime, timedelta


# -------------------- APP SETUP --------------------
app = Flask(__name__)

model = load_model('model/weather_lstm.h5', compile=False)
scaler = joblib.load('model/scaler.save')

OPENWEATHER_API_KEY = "375ffecb477f6563fb6ccc78021416e1"
WEATHERAPI_KEY = "9a849ef7d2bb42a0b22202436262504"

SEQ_LENGTH = 10


# -------------------- ROUTE --------------------
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        city = request.form.get('city')

        # -------- WEATHER (OpenWeather) --------
        try:
            url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
            data = requests.get(url, timeout=5).json()
        except:
            return render_template('index.html', error="API connection error")

        if str(data.get("cod")) != "200":
            return render_template('index.html', error=data.get("message", "Invalid city"))

        temps, dates = [], []

        for item in data['list'][:SEQ_LENGTH]:
            temps.append(item['main']['temp'])
            dt = datetime.fromtimestamp(item['dt'])
            dates.append(dt.strftime("%d %b"))

        # -------- CURRENT WEATHER --------
        current = data['list'][0]
        current_temp = current['main']['temp']
        humidity = current['main']['humidity']
        wind_speed = round(current['wind']['speed'] * 3.6, 1)

        # -------- COORDINATES --------
        lat = data['city']['coord']['lat']
        lon = data['city']['coord']['lon']

        # -------- AQI --------
        try:
            aqi_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
            aqi_data = requests.get(aqi_url, timeout=5).json()
            aqi = aqi_data.get('list', [{}])[0].get('main', {}).get('aqi', 0)

            aqi_status_list = ["Good", "Fair", "Moderate", "Poor", "Very Poor"]
            aqi_status = aqi_status_list[aqi - 1] if aqi in range(1, 6) else "N/A"
        except:
            aqi, aqi_status = 0, "N/A"

        # -------- UV INDEX (WeatherAPI) --------
        try:
            uv_url = f"http://api.weatherapi.com/v1/current.json?key={WEATHERAPI_KEY}&q={city}"
            uv_data = requests.get(uv_url, timeout=5).json()

            uv_index = round(uv_data.get("current", {}).get("uv", 0), 1)
        except:
            uv_index = 0

        # -------- UV LABEL --------
        if uv_index <= 2:
            uv_label = "Low"
        elif uv_index <= 5:
            uv_label = "Moderate"
        elif uv_index <= 7:
            uv_label = "High"
        elif uv_index <= 10:
            uv_label = "Very High"
        else:
            uv_label = "Extreme"

        # -------- MODEL PREDICTION --------
        input_data = np.array(temps).reshape(-1, 1)
        input_scaled = scaler.transform(input_data)

        future_preds = []
        current_seq = input_scaled.reshape(1, SEQ_LENGTH, 1)

        for _ in range(3):
            pred = model.predict(current_seq, verbose=0)
            future_preds.append(pred[0][0])

            new_seq = np.append(current_seq[0][1:], pred, axis=0)
            current_seq = new_seq.reshape(1, SEQ_LENGTH, 1)

        future_preds = scaler.inverse_transform(
            np.array(future_preds).reshape(-1, 1)
        )

        future_dates = [
            (datetime.now() + timedelta(days=i)).strftime("%d %b")
            for i in range(1, 4)
        ]

        return render_template(
            'index.html',
            city=city,
            dates=dates,
            temps=temps,
            future_dates=future_dates,
            future_preds=[round(float(x[0]), 2) for x in future_preds],
            current_temp=round(current_temp, 1),
            humidity=humidity,
            wind_speed=wind_speed,
            uv_index=uv_index,
            uv_label=uv_label,
            aqi=aqi,
            aqi_status=aqi_status
        )

    return render_template('index.html')


# -------------------- RUN --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)