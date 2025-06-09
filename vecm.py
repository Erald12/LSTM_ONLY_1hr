import pandas as pd
import numpy as np
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
    'BTC_CLOSE': raw_data3['BTC_CLOSE'],
    'BTC_CLOSE_SHIFTED': raw_data3['BTC_CLOSE'].shift(564),
    'TOTAL3_CLOSE': raw_data3['TOTAL3_CLOSE'],
    'TOTAL3_CLOSE_SHIFTED': raw_data3['TOTAL3_CLOSE'].shift(564)
}).dropna()

main_data = pd.DataFrame({
    'XRPUSD_CLOSE': main_data0['XRPUSD_CLOSE'],
    'DXY_CLOSE': main_data0['DXY_CLOSE'],
    'BTC_CLOSE': main_data0['BTC_CLOSE'],
    'BTC_CLOSE_SHIFTED': main_data0['BTC_CLOSE_SHIFTED'],
    'TOTAL3_CLOSE': main_data0['TOTAL3_CLOSE'],
    'TOTAL3_CLOSE_SHIFTED': main_data0['TOTAL3_CLOSE_SHIFTED'],
    'XRP_BTC_CORR': compute_rolling_correlation(main_data0['XRPUSD_CLOSE'],main_data0['BTC_CLOSE_SHIFTED'],20),
    'XRP_TOTAL3_CORR': compute_rolling_correlation(main_data0['XRPUSD_CLOSE'],main_data0['TOTAL3_CLOSE_SHIFTED'],20),
}).dropna()

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

laggs = 177
# STEP 1: Create lagged independent variables for cointegration test
main_data['BTC_L1'] = main_data['BTC_CLOSE'].shift(laggs)
main_data['TOTAL3_L1'] = main_data['XRP_TOTAL3_CORR'].shift(laggs)
main_data = main_data.dropna()

# STEP 2: Cointegration regression (Engle-Granger Step 1)
X = sm.add_constant(main_data[['BTC_L1']])
y = main_data['XRPUSD_CLOSE']
model = sm.OLS(y, X).fit()
main_data['residual'] = model.resid

# STEP 3: Test for stationarity of residuals (Engle-Granger Step 2)
adf_result = adfuller(main_data['residual'])
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')
print(model.summary())

from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

def find_best_lag_for_cointegration(df, max_lag=700):
    results = []

    for lag in range(0, max_lag + 1):
        df_lagged = df.copy()
        df_lagged['XRP_BTC_LAG'] = df_lagged['XRP_BTC_CORR'].shift(lag)
        df_lagged['XRP_TOTAL3_LAG'] = df_lagged['XRP_TOTAL3_CORR'].shift(lag)
        df_lagged['BTC_SHIFTED'] = df_lagged['BTC_CLOSE'].shift(lag)
        df_lagged['XRP_SHIFTED'] = df_lagged['XRPUSD_CLOSE'].shift(lag)
        df_lagged['TOTAL3_SHIFTED'] = df_lagged['TOTAL3_CLOSE'].shift(lag)
        df_lagged = df_lagged.dropna()

        X = df_lagged[['BTC_SHIFTED']] #38
        X = sm.add_constant(X)
        y = df_lagged['XRPUSD_CLOSE']

        model = sm.OLS(y, X).fit()
        residuals = model.resid
        adf_stat, pvalue, *_ = adfuller(residuals)

        results.append({
            'lag': lag,
            'ADF_stat': adf_stat,
            'p-value': pvalue
        })

    return pd.DataFrame(results).sort_values('ADF_stat')

# Example usage:
best_lag_df = find_best_lag_for_cointegration(main_data, max_lag=700)
print(best_lag_df)

# Save the results DataFrame to a CSV file
best_lag_df.to_csv('best_lag_results.csv', index=False)

print("Results have been saved to 'best_lag_results.csv'")

