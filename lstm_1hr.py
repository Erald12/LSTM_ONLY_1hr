import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Reshape, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator, SMAIndicator, CCIIndicator
from ta.momentum import RSIIndicator, ROCIndicator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer


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

# Scaling data
scaler = MinMaxScaler()
main_data_scaled = scaler.fit_transform(selected_data1)
joblib.dump(scaler, 'scaler.pkl')

# Sequence creation
feature_indices = [4]  # assuming column 4 is the target
def create_sequence(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len, feature_indices])
        y.append(data[i+seq_len, [4]])
    return np.array(X), np.array(y)

sequence_length = 20
X, y = create_sequence(main_data_scaled, sequence_length)

# Split train/test
test_size = 610
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

# --- Model Architecture ---
from custom_objects2 import heteroscedastic_loss, ClipByValueLayer

sequence_input = Input(shape=(sequence_length, X.shape[2]), name="sequence_input")

lstm_1 = LSTM(
    100, return_sequences=True, activation='tanh',
    kernel_regularizer=regularizers.l2(0.001),
    recurrent_regularizer=regularizers.l2(0.001),
    bias_regularizer=regularizers.l2(0.001)
)(sequence_input)
lstm_1 = Dropout(0.3)(lstm_1)

lstm_2 = LSTM(
    100, return_sequences=False, activation='tanh',
    kernel_regularizer=regularizers.l2(0.001),
    recurrent_regularizer=regularizers.l2(0.001),
    bias_regularizer=regularizers.l2(0.001)
)(lstm_1)
lstm_2 = Dropout(0.3)(lstm_2)

dense_1 = Dense(100, activation='swish')(lstm_2)
dense_1 = Dropout(0.3)(dense_1)

mu_logvar = Dense(2, activation='linear')(dense_1)

# Custom layers for mu and log_var
mu, log_var = SplitMuLogVar()(mu_logvar)
log_var = ClipByValueLayer(-5.0, 5.0)(log_var)

# Combine outputs without reshaping
output = Concatenate(axis=1)([mu, log_var])  # Shape: [batch_size, 2]

# Model compilation
model2 = Model(inputs=sequence_input, outputs=output)
optimizer = Adam(learning_rate=0.00008, clipnorm=1.0)
model2.compile(optimizer=optimizer, loss=heteroscedastic_loss)

# Custom Model Checkpoint callback
class CustomModelCheckpoint(Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.best_val_loss = np.inf
        self.best_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get("val_loss")
        current_loss = logs.get("loss")

        if current_val_loss is None or current_loss is None:
            return

        # Save if val_loss improves
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_loss = current_loss
            self.model.save(self.filepath)
            print(f"Epoch {epoch + 1}: val_loss improved to {current_val_loss:.4f}. Model saved.")
        # If val_loss is same but loss improves
        elif current_val_loss == self.best_val_loss and current_loss < self.best_loss:
            self.best_loss = current_loss
            self.model.save(self.filepath)
            print(f"Epoch {epoch + 1}: val_loss tied but loss improved to {current_loss:.4f}. Model saved.")

# EarlyStopping callback
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=90,
    restore_best_weights=True,
    verbose=1
)

# Model training with callbacks
checkpoint_path = "best_model.keras"
custom_checkpoint = CustomModelCheckpoint(checkpoint_path)

model2.fit(
    X_train, y_train,  # Pass both sequence and condition inputs
    epochs=500,
    batch_size=8,
    validation_data=(X_test, y_test),  # Pass both sequence and condition inputs for validation
    callbacks=[custom_checkpoint, early_stop])
