import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # remove TensorFlow warning

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import joblib

# =========================
# 1. LOAD DATA
# =========================
data = pd.read_csv('ghcn_daily_1901_kaggle.csv')

print("Original Columns:", data.columns)

# =========================
# 2. CONVERT LONG → WIDE
# =========================
data = data.pivot_table(index='date', columns='element', values='value')

# Reset index
data.reset_index(inplace=True)

print("After Pivot Columns:", data.columns)

# =========================
# 3. CHECK REQUIRED COLUMNS
# =========================
if 'TMAX' not in data.columns or 'TMIN' not in data.columns:
    raise ValueError("Dataset must contain TMAX and TMIN")

# =========================
# 4. CLEAN DATA
# =========================

# Convert to Celsius (NOAA stores *10)
data['TMAX'] = data['TMAX'] / 10
data['TMIN'] = data['TMIN'] / 10

# Keep only required columns
data = data[['TMAX', 'TMIN']]

# Drop rows where both are missing
data = data.dropna(subset=['TMAX', 'TMIN'])

print("After Cleaning Shape:", data.shape)

# =========================
# 5. CREATE FEATURE
# =========================
data['TEMP'] = (data['TMAX'] + data['TMIN']) / 2

temp = data['TEMP'].values.reshape(-1, 1)

# Safety check
if len(temp) == 0:
    raise ValueError("❌ Dataset is empty after preprocessing")

# =========================
# 6. NORMALIZATION
# =========================
scaler = MinMaxScaler()
temp_scaled = scaler.fit_transform(temp)

# =========================
# 7. CREATE SEQUENCES
# =========================
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(temp_scaled)

print("Sequence Shape:", X.shape)

# =========================
# 8. TRAIN TEST SPLIT
# =========================
split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =========================
# 9. BUILD LSTM MODEL
# =========================
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mae')

# =========================
# 10. TRAIN MODEL
# =========================
model.fit(X_train, y_train, epochs=5, batch_size=32)

# =========================
# 11. SAVE MODEL
# =========================
model.save('model/weather_lstm.h5')
joblib.dump(scaler, 'model/scaler.save')

print("✅ Model trained and saved successfully!")