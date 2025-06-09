import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import ccf


import ta  # Technical Analysis library


def compute_rolling_correlation(series1, series2, window):
    """
    Computes the rolling Pearson correlation between two series.

    Parameters:
        series1 (pd.Series): First input time series.
        series2 (pd.Series): Second input time series.
        window (int): Rolling window size. Default is 14.

    Returns:
        pd.Series: Rolling correlation values.
    """
    return series1.rolling(window).corr(series2)

# --- RSI Function ---
def calculate_rsi(close_prices, period=14):
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_rsi14(close_series):
    rsi_indicator = ta.momentum.RSIIndicator(close=close_series, window=14)
    return rsi_indicator.rsi()

def compute_ema(close_series,period):
    ema_indicator = ta.trend.EMAIndicator(close=close_series, window=period)
    return ema_indicator.ema_indicator()

def compute_sma(close_series, period):
    sma_indicator = ta.trend.SMAIndicator(close=close_series, window=period)
    return sma_indicator.sma_indicator()

def bounded_cci(series, window=20, clip_min=-200, clip_max=200):
    # Calculate CCI (unbounded)
    cci = ta.trend.CCIIndicator(high=series, low=series, close=series, window=window).cci()
    # Clip to avoid extreme outliers
    #cci_clipped = np.clip(cci, clip_min, clip_max)
    # Apply tanh after clipping to bound between -1 and 1
    return np.tanh(cci / 100)

def bounded_roc(series, window=10, clip_min=-30, clip_max=30):
    # Calculate ROC (unbounded)
    roc = ta.momentum.ROCIndicator(close=series, window=window).roc()
    # Clip to avoid extreme outliers
    #roc_clipped = np.clip(roc, clip_min, clip_max)
    # Apply tanh after clipping to bound between -1 and 1
    return np.tanh(roc / 10)

# Load data
raw_data = pd.read_csv('main_data.csv', index_col='datetime')

raw_data2 = pd.DataFrame({
    'XRPUSD_CLOSE': raw_data['XRPUSD_CLOSE'],
    'XRP_HIGH_CLOSE_DIFF': raw_data['XRP_MC_HIGH_CLOSE_DIFF'],
    'XRP_LOW_CLOSE_DIFF': raw_data['XRP_MC_LOW_CLOSE_DIFF'],
    'TOTAL_CLOSE': raw_data['TOTAL_MC_CLOSE'],
    'TOTAL_HIGH_CLOSE_DIFF': raw_data['TOTAL_MC_HIGH_CLOSE_DIFF'],
    'TOTAL_LOW_CLOSE_DIFF': raw_data['TOTAL_MC_LOW_CLOSE_DIFF'],
    'BTC_CLOSE': raw_data['BTC_MC_CLOSE'],
    'BTC_HIGH_CLOSE_DIFF': raw_data['BTC_MC_HIGH_CLOSE_DIFF'],
    'BTC_LOW_CLOSE_DIFF': raw_data['BTC_MC_LOW_CLOSE_DIFF'],
    'DXY_CLOSE': raw_data['DXY_CLOSE'],
    'DXY_HIGH_CLOSE_DIFF': raw_data['DXY_HIGH_CLOSE_DIFF'],
    'DXY_LOW_CLOSE_DIFF': raw_data['DXY_LOW_CLOSE_DIFF'],
    'TOTAL2_CLOSE': raw_data['TOTAL2_CLOSE'],
    'TOTAL2_HIGH_CLOSE_DIFF': raw_data['TOTAL2_HIGH_CLOSE_DIFF'],
    'TOTAL2_LOW_CLOSE_DIFF': raw_data['TOTAL2_LOW_CLOSE_DIFF'],
    'TOTAL3_CLOSE': raw_data['TOTAL3_CLOSE'],
    'TOTAL3_HIGH_CLOSE_DIFF': raw_data['TOTAL3_HIGH_CLOSE_DIFF'],
    'TOTAL3_LOW_CLOSE_DIFF': raw_data['TOTAL3_LOW_CLOSE_DIFF']
})



raw_data3 = raw_data2

main_data0 = pd.DataFrame({
    'XRPUSD_CLOSE': raw_data3['XRPUSD_CLOSE'],
    'DXY_CLOSE': raw_data3['DXY_CLOSE'],
    'BTC_CLOSE_SHIFTED': raw_data3['BTC_CLOSE'].shift(564),
    'TOTAL3_CLOSE_SHIFTED': raw_data3['TOTAL3_CLOSE'].shift(564)
}).dropna()

main_data = pd.DataFrame({
    'XRPUSD_CLOSE': main_data0['XRPUSD_CLOSE'],
    'XRP_BTC_CORR': compute_rolling_correlation(main_data0['XRPUSD_CLOSE'],main_data0['BTC_CLOSE_SHIFTED'],20),
    'XRP_TOTAL3_CORR': compute_rolling_correlation(main_data0['XRPUSD_CLOSE'],main_data0['TOTAL3_CLOSE_SHIFTED'],20),
}).dropna()


from statsmodels.tsa.stattools import adfuller
print(adfuller(main_data['XRP_BTC_CORR']))
print(adfuller(main_data['XRP_TOTAL3_CORR']))


from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

result = ar_select_order(main_data['XRP_BTC_CORR'], maxlag=100, ic='aic')
print(result.ar_lags)

from statsmodels.tsa.stattools import grangercausalitytests

# The test expects the columns in the order [Y, X] – testing if X causes Y
max_lag = 100  # try a few values like 1 to 5


def granger_causality_table(data, variables, max_lag):
    """
    Returns a DataFrame with F-test statistics, p-values, and correlation
    between the dependent variable and the lagged independent variable.
    """
    results = {}
    test_result = grangercausalitytests(data[variables], maxlag=max_lag, verbose=False)

    target_var = variables[0]
    other_var = variables[1]

    for lag in range(1, max_lag + 1):
        f_test = test_result[lag][0]['ssr_ftest']

        # Create lagged version of the independent variable
        lagged_col = data[other_var].shift(lag)

        # Drop NaNs due to shifting
        valid_data = pd.DataFrame({
            target_var: data[target_var],
            f'{other_var}_lag{lag}': lagged_col
        }).dropna()

        # Compute correlation
        correlation = valid_data[target_var].corr(valid_data[f'{other_var}_lag{lag}'])

        results[lag] = {
            'F-statistic': f_test[0],
            'p-value': f_test[1],
            'Correlation': correlation
        }

    return pd.DataFrame(results).T


print('=====================BTC=======================')
result = granger_causality_table(main_data, ['XRP_BTC_CORR', 'XRP_TOTAL3_CORR'], max_lag)
print(result)

min_p_value = result['p-value'].min()
best_lag = result['p-value'].idxmin()

print(f"Lowest p-value: {min_p_value:.6f} at lag {best_lag}")

significant_lags = result[result['p-value'] < 0.01]
print(significant_lags)

print("Significant lags (p < 0.01):", significant_lags.index.tolist())





