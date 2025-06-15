import pandas as pd
import numpy as np
from scipy.stats import norm
import joblib
from tensorflow.keras.models import load_model
from keras import config
import tensorflow as tf
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator, SMAIndicator, CCIIndicator
from ta.momentum import RSIIndicator, ROCIndicator
from tensorflow.keras.layers import Layer

from custom_objects2 import (
    heteroscedastic_loss,
    ClipByValueLayer
)

# Enable unsafe deserialization for custom objects in model loading
config.enable_unsafe_deserialization()

# --- Custom Functions ---
def compute_rolling_correlation(series1, series2, window):
    return series1.rolling(window).corr(series2)

def calculate_rsi(close_prices, period=14):
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bollinger_bands(close_series, period=20, std_multiplier=2):
    bb_indicator = BollingerBands(close=close_series, window=period, window_dev=std_multiplier)
    middle_band = bb_indicator.bollinger_mavg()
    upper_band = bb_indicator.bollinger_hband()
    lower_band = bb_indicator.bollinger_lband()
    diff = (upper_band - middle_band) / 2
    return middle_band, diff

def compute_rsi14(close_series):
    return RSIIndicator(close=close_series, window=14).rsi()

def compute_ema(close_series, period):
    return EMAIndicator(close=close_series, window=period).ema_indicator()

def compute_sma(close_series, period):
    return SMAIndicator(close=close_series, window=period).sma_indicator()

def bounded_cci(series, window=20):
    cci = CCIIndicator(high=series, low=series, close=series, window=window).cci()
    return np.tanh(cci / 100)

def bounded_roc(series, window=10):
    roc = ROCIndicator(close=series, window=window).roc()
    return np.tanh(roc / 10)

class SplitMuLogVar(Layer):
    def call(self, inputs):
        mu = inputs[:, :1]
        log_var = inputs[:, 1:]
        return mu, log_var

# --- Load and Process Data ---

raw_data = pd.read_csv('main_data.csv', index_col='datetime')

raw_data2 = pd.DataFrame({
    'XRPUSD_OPEN': raw_data['XRPUSD_OPEN'],
    'XRPUSD_CLOSE': raw_data['XRPUSD_CLOSE'],
    'XRPUSD_LOW': raw_data['XRPUSD_LOW'],
    'XRPUSD_HIGH': raw_data['XRPUSD_HIGH'],
    'XRP_HIGH_CLOSE_DIFF': raw_data['XRP_MC_HIGH_CLOSE_DIFF'],
    'XRP_LOW_CLOSE_DIFF': raw_data['XRP_MC_LOW_CLOSE_DIFF'],
    'TOTAL_CLOSE': raw_data['TOTAL_MC_CLOSE'],
    'TOTAL_LOW': raw_data['TOTAL_MC_LOW'],
    'TOTAL_HIGH': raw_data['TOTAL_MC_HIGH'],
    'TOTAL_HIGH_CLOSE_DIFF': raw_data['TOTAL_MC_HIGH_CLOSE_DIFF'],
    'TOTAL_LOW_CLOSE_DIFF': raw_data['TOTAL_MC_LOW_CLOSE_DIFF'],
    'BTC_CLOSE': raw_data['BTC_MC_CLOSE'],
    'BTC_LOW': raw_data['BTC_MC_LOW'],
    'BTC_HIGH': raw_data['BTC_MC_HIGH'],
    'BTC_HIGH_CLOSE_DIFF': raw_data['BTC_MC_HIGH_CLOSE_DIFF'],
    'BTC_LOW_CLOSE_DIFF': raw_data['BTC_MC_LOW_CLOSE_DIFF'],
    'DXY_CLOSE': raw_data['DXY_CLOSE'],
    'DXY_HIGH_CLOSE_DIFF': raw_data['DXY_HIGH_CLOSE_DIFF'],
    'DXY_LOW_CLOSE_DIFF': raw_data['DXY_LOW_CLOSE_DIFF'],
    'TOTAL2_CLOSE': raw_data['TOTAL2_CLOSE'],
    'TOTAL2_HIGH_CLOSE_DIFF': raw_data['TOTAL2_HIGH_CLOSE_DIFF'],
    'TOTAL2_LOW_CLOSE_DIFF': raw_data['TOTAL2_LOW_CLOSE_DIFF'],
    'TOTAL3_CLOSE': raw_data['TOTAL3_CLOSE'],
    'TOTAL3_LOW': raw_data['TOTAL3_LOW'],
    'TOTAL3_HIGH': raw_data['TOTAL3_HIGH'],
    'TOTAL3_HIGH_CLOSE_DIFF': raw_data['TOTAL3_HIGH_CLOSE_DIFF'],
    'TOTAL3_LOW_CLOSE_DIFF': raw_data['TOTAL3_LOW_CLOSE_DIFF']
}).dropna()

# Selecting relevant columns for training
raw_data3 = raw_data2[['XRPUSD_CLOSE','XRPUSD_LOW','XRPUSD_HIGH', 'DXY_CLOSE', 'BTC_CLOSE','BTC_LOW','BTC_HIGH', 'TOTAL_CLOSE', 'TOTAL3_CLOSE','TOTAL3_LOW','TOTAL3_HIGH']]

# Create lagged feature
selected_data0 = pd.DataFrame({
    'XRPUSD_CLOSE': raw_data3['XRPUSD_CLOSE'],
    'BTC_CLOSE': raw_data3['BTC_CLOSE'],
    'BTC_CLOSE_SHIFTED': raw_data3['BTC_CLOSE'].shift(335),
    'BTC_CLOSE_OFFSET': raw_data3['BTC_CLOSE'].shift(335-99),
    'DXY_CLOSE': raw_data3['DXY_CLOSE'],
    'TOTAL3_CLOSE': raw_data3['TOTAL3_CLOSE']
}).dropna()

selected_data1 = pd.DataFrame({
    'XRPUSD_CLOSE': selected_data0['XRPUSD_CLOSE'],
    'BTC_CLOSE_OFFSET': selected_data0['BTC_CLOSE_OFFSET'],
    'DXY_CLOSE': selected_data0['DXY_CLOSE'],
    'TOTAL3_CLOSE': selected_data0['TOTAL3_CLOSE'],
    'XRPUSD_BTC_CORRELATION': compute_rolling_correlation(selected_data0['XRPUSD_CLOSE'], selected_data0['BTC_CLOSE_SHIFTED'], 33)
}).dropna()

# Load scaler and scale data
scaler = joblib.load('scaler.pkl')
main_data_scaled = scaler.fit_transform(selected_data1)


# Sequence creation
feature_indices = [4]  # assuming column 4 is the target
def create_sequence(data, seq_len):
    X, y, z = [], [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len, feature_indices])
        y.append(data[i+seq_len, [4]])
        z.append(data[i+seq_len, [1]])
    return np.array(X), np.array(y), np.array(z)

sequence_length = 20
X, y, z = create_sequence(main_data_scaled, sequence_length)

# Load model with custom objects
model = load_model('best_model.keras', custom_objects={
    'heteroscedastic_loss': heteroscedastic_loss,
    'ClipByValueLayer': ClipByValueLayer,
    'SplitMuLogVar':SplitMuLogVar,
}, compile=False)

# Predict
predictions = model.predict(X)

# Start from the last known sequence
last_sequence = X[-1]  # shape: (sequence_length, 1) if 1 feature
recursive_forecast_mean = []
recursive_forecast_std = []


current_sequence = last_sequence.copy()

forecast_length = 22

for _ in range(forecast_length):
    # Predict the next value
    pred = model.predict(current_sequence[np.newaxis, ...])  # shape: (1, 1)

    # If your model outputs mean and log_var (like in Bayesian models), get mean only
    if pred.shape[-1] == 2:
        pred_value = pred[0, 0]  # assuming this is the mean (mu)
        pred_value_std = pred[0, 1]
    else:
        pred_value = pred[0, 0]
        pred_value_std = pred[0, 1]

    # Save the prediction
    recursive_forecast_mean.append(pred_value)
    recursive_forecast_std.append(pred_value_std)

    # Update the sequence with the new predicted value
    new_step = np.array([[pred_value]])  # shape (1, 1)
    current_sequence = np.vstack([current_sequence[1:], new_step])  # remove oldest, add newest


recursive_forecast_mean = np.array(recursive_forecast_mean)
recursive_forecast_std = np.array(recursive_forecast_std)

def inverse_transform_single_feature(scaled_array, feature_index):
    scaled_array = np.array(scaled_array).reshape(-1)  # ensure 1D array
    dummy = np.zeros((scaled_array.shape[0], scaler.scale_.shape[0]))
    dummy[:, feature_index] = scaled_array
    inv = scaler.inverse_transform(dummy)
    return inv[:, feature_index]

y_inv = inverse_transform_single_feature(y, 4)
z_inv = inverse_transform_single_feature(z, 1)

combined_actual_forecast = np.concatenate((predictions[:, 0],recursive_forecast_mean))
combined_actual_forecast_std = np.concatenate((predictions[:, 1],recursive_forecast_std))

combined_actual_forecast_inverse = inverse_transform_single_feature(combined_actual_forecast,4)
combined_actual_forecast_inverse_std = inverse_transform_single_feature(combined_actual_forecast_std,4)

# Compute confidence bounds
upper_bound = combined_actual_forecast_inverse + combined_actual_forecast_inverse_std
lower_bound = combined_actual_forecast_inverse - combined_actual_forecast_inverse_std


import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

# --- Subplot 1: Actual vs Predicted/Forecast Correlation ---
ax1.plot(y_inv, label='Actual Correlation (y_inv)', color='blue')
ax1.plot(combined_actual_forecast_inverse, label='Predicted & Forecast Correlation', color='orange')
# Add optional forecast start marker
forecast_start = len(y_inv)
ax1.axvline(x=forecast_start, color='gray', linestyle='--', label='Forecast Start')
ax1.set_title('Actual vs Predicted/Forecast Correlation Between XRP and BTC lag 335')
ax1.set_ylabel('Correlation (Bounded)')
ax1.legend()
ax1.grid(True)

# --- Subplot 2: BTC Offset (z_inv) ---
ax2.plot(z_inv, label='BTC Offset (z_inv)', color='green')

# Add vertical line at 99 candles before the end
vline_position = len(z_inv) - 99
vline_position2 = len(z_inv) - 99 + 33
vline_position3 = len(z_inv) - 99 + 33 + 33
ax2.axvline(x=vline_position, color='red', linestyle='--', label='Last 99 candles start')
ax2.axvline(x=vline_position2, color='blue', linestyle='--', label='Last 66 candles start')
ax2.axvline(x=vline_position3, color='black', linestyle='--', label='Last 33 candles start')

#ax2.set_title('BTC Offset Feature (z_inv)')
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('BTC Close Offset (Inverse Scaled)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
