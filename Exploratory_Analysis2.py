import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import ccf

raw_data = pd.read_csv('main_data.csv', index_col='datetime')

raw_data2 = pd.DataFrame({
    'XRPUSD_CLOSE':raw_data['XRPUSD_CLOSE'],
    'XRP_HIGH_CLOSE_DIFF': raw_data['XRP_MC_HIGH_CLOSE_DIFF'],
    'XRP_LOW_CLOSE_DIFF': raw_data['XRP_MC_LOW_CLOSE_DIFF'],
    'TOTAL_CLOSE': raw_data['TOTAL_MC_CLOSE'],
    'TOTAL_HIGH_CLOSE_DIFF':raw_data['TOTAL_MC_HIGH_CLOSE_DIFF'],
    'TOTAL_LOW_CLOSE_DIFF':raw_data['TOTAL_MC_LOW_CLOSE_DIFF'],
    'BTC_CLOSE':raw_data['BTC_MC_CLOSE'],
    'BTC_HIGH_CLOSE_DIFF':raw_data['BTC_MC_HIGH_CLOSE_DIFF'],
    'BTC_LOW_CLOSE_DIFF':raw_data['BTC_MC_LOW_CLOSE_DIFF'],
    'DXY_CLOSE':raw_data['DXY_CLOSE'],
    'DXY_HIGH_CLOSE_DIFF':raw_data['DXY_HIGH_CLOSE_DIFF'],
    'DXY_LOW_CLOSE_DIFF':raw_data['DXY_LOW_CLOSE_DIFF'],
    'TOTAL2_CLOSE':raw_data['TOTAL2_CLOSE'],
    'TOTAL2_HIGH_CLOSE_DIFF':raw_data['TOTAL2_HIGH_CLOSE_DIFF'],
    'TOTAL2_LOW_CLOSE_DIFF':raw_data['TOTAL2_LOW_CLOSE_DIFF'],
    'TOTAL3_CLOSE':raw_data['TOTAL3_CLOSE'],
    'TOTAL3_HIGH_CLOSE_DIFF':raw_data['TOTAL3_HIGH_CLOSE_DIFF'],
    'TOTAL3_LOW_CLOSE_DIFF':raw_data['TOTAL3_LOW_CLOSE_DIFF']
})

raw_data3 = raw_data2

#Compute SMA's
raw_data3['TOTAL_CLOSE_SMA9'] = raw_data3['TOTAL_CLOSE'].rolling(window=9).mean()
raw_data3['TOTAL_CLOSE_SMA20'] = raw_data3['TOTAL_CLOSE'].rolling(window=20).mean()
raw_data3['TOTAL_CLOSE_SMA50'] = raw_data3['TOTAL_CLOSE'].rolling(window=50).mean()
raw_data3['TOTAL_CLOSE_SMA100'] = raw_data3['TOTAL_CLOSE'].rolling(window=100).mean()

raw_data3['BTC_CLOSE_SMA9'] = raw_data3['BTC_CLOSE'].rolling(window=9).mean()
raw_data3['BTC_CLOSE_SMA20'] = raw_data3['BTC_CLOSE'].rolling(window=20).mean()
raw_data3['BTC_CLOSE_SMA50'] = raw_data3['BTC_CLOSE'].rolling(window=50).mean()
raw_data3['BTC_CLOSE_SMA100'] = raw_data3['BTC_CLOSE'].rolling(window=100).mean()

raw_data3['TOTAL2_CLOSE_SMA9'] = raw_data3['TOTAL2_CLOSE'].rolling(window=9).mean()
raw_data3['TOTAL2_CLOSE_SMA20'] = raw_data3['TOTAL2_CLOSE'].rolling(window=20).mean()
raw_data3['TOTAL2_CLOSE_SMA50'] = raw_data3['TOTAL2_CLOSE'].rolling(window=50).mean()
raw_data3['TOTAL2_CLOSE_SMA100'] = raw_data3['TOTAL2_CLOSE'].rolling(window=100).mean()

raw_data3['TOTAL3_CLOSE_SMA9'] = raw_data3['TOTAL3_CLOSE'].rolling(window=9).mean()
raw_data3['TOTAL3_CLOSE_SMA20'] = raw_data3['TOTAL3_CLOSE'].rolling(window=20).mean()
raw_data3['TOTAL3_CLOSE_SMA50'] = raw_data3['TOTAL3_CLOSE'].rolling(window=50).mean()
raw_data3['TOTAL3_CLOSE_SMA100'] = raw_data3['TOTAL3_CLOSE'].rolling(window=100).mean()

raw_data3['XRPUSD_CLOSE_SMA9'] = raw_data3['XRPUSD_CLOSE'].rolling(window=9).mean()
raw_data3['XRPUSD_CLOSE_SMA20'] = raw_data3['XRPUSD_CLOSE'].rolling(window=20).mean()
raw_data3['XRPUSD_CLOSE_SMA50'] = raw_data3['XRPUSD_CLOSE'].rolling(window=50).mean()
raw_data3['XRPUSD_CLOSE_SMA100'] = raw_data3['XRPUSD_CLOSE'].rolling(window=100).mean()

raw_data3.dropna(inplace=True)

#Calculate for Mean Reversion Measure
raw_data3['MEAN9_REV_TOTAL_CLOSE'] = raw_data3['TOTAL_CLOSE'] - raw_data3['TOTAL_CLOSE_SMA9']
raw_data3['MEAN20_REV_TOTAL_CLOSE'] = raw_data3['TOTAL_CLOSE'] - raw_data3['TOTAL_CLOSE_SMA20']
raw_data3['MEAN50_REV_TOTAL_CLOSE'] = raw_data3['TOTAL_CLOSE'] - raw_data3['TOTAL_CLOSE_SMA50']
raw_data3['MEAN100_REV_TOTAL_CLOSE'] = raw_data3['TOTAL_CLOSE'] - raw_data3['TOTAL_CLOSE_SMA100']

raw_data3['MEAN9_REV_BTC_CLOSE'] = raw_data3['BTC_CLOSE'] - raw_data3['BTC_CLOSE_SMA9']
raw_data3['MEAN20_REV_BTC_CLOSE'] = raw_data3['BTC_CLOSE'] - raw_data3['BTC_CLOSE_SMA20']
raw_data3['MEAN50_REV_BTC_CLOSE'] = raw_data3['BTC_CLOSE'] - raw_data3['BTC_CLOSE_SMA50']
raw_data3['MEAN100_REV_BTC_CLOSE'] = raw_data3['BTC_CLOSE'] - raw_data3['BTC_CLOSE_SMA100']

raw_data3['MEAN9_REV_TOTAL2_CLOSE'] = raw_data3['TOTAL2_CLOSE'] - raw_data3['TOTAL2_CLOSE_SMA9']
raw_data3['MEAN20_REV_TOTAL2_CLOSE'] = raw_data3['TOTAL2_CLOSE'] - raw_data3['TOTAL2_CLOSE_SMA20']
raw_data3['MEAN50_REV_TOTAL2_CLOSE'] = raw_data3['TOTAL2_CLOSE'] - raw_data3['TOTAL2_CLOSE_SMA50']
raw_data3['MEAN100_REV_TOTAL2_CLOSE'] = raw_data3['TOTAL2_CLOSE'] - raw_data3['TOTAL2_CLOSE_SMA100']

raw_data3['MEAN9_REV_TOTAL3_CLOSE'] = raw_data3['TOTAL3_CLOSE'] - raw_data3['TOTAL3_CLOSE_SMA9']
raw_data3['MEAN20_REV_TOTAL3_CLOSE'] = raw_data3['TOTAL3_CLOSE'] - raw_data3['TOTAL3_CLOSE_SMA20']
raw_data3['MEAN50_REV_TOTAL3_CLOSE'] = raw_data3['TOTAL3_CLOSE'] - raw_data3['TOTAL3_CLOSE_SMA50']
raw_data3['MEAN100_REV_TOTAL3_CLOSE'] = raw_data3['TOTAL3_CLOSE'] - raw_data3['TOTAL3_CLOSE_SMA100']

raw_data3['MEAN9_REV_XRPUSD_CLOSE'] = raw_data3['XRPUSD_CLOSE'] - raw_data3['XRPUSD_CLOSE_SMA9']
raw_data3['MEAN20_REV_XRPUSD_CLOSE'] = raw_data3['XRPUSD_CLOSE'] - raw_data3['XRPUSD_CLOSE_SMA20']
raw_data3['MEAN50_REV_XRPUSD_CLOSE'] = raw_data3['XRPUSD_CLOSE'] - raw_data3['XRPUSD_CLOSE_SMA50']
raw_data3['MEAN100_REV_XRPUSD_CLOSE'] = raw_data3['XRPUSD_CLOSE'] - raw_data3['XRPUSD_CLOSE_SMA100']

raw_data3['BTC_DOM'] = raw_data3['BTC_CLOSE']/raw_data3['TOTAL_CLOSE']

main_data = pd.DataFrame({
    'XRPUSD_CLOSE':raw_data3['XRPUSD_CLOSE'],
    'XRP_HIGH_CLOSE_DIFF': raw_data3['XRP_HIGH_CLOSE_DIFF'],
    'XRP_LOW_CLOSE_DIFF': raw_data3['XRP_LOW_CLOSE_DIFF'],
    'TOTAL_CLOSE': raw_data3['TOTAL_CLOSE'],
    'TOTAL_HIGH_CLOSE_DIFF':raw_data3['TOTAL_HIGH_CLOSE_DIFF'],
    'TOTAL_LOW_CLOSE_DIFF':raw_data3['TOTAL_LOW_CLOSE_DIFF'],
    'BTC_CLOSE':raw_data3['BTC_CLOSE'],
    'BTC_HIGH_CLOSE_DIFF':raw_data3['BTC_HIGH_CLOSE_DIFF'],
    'BTC_LOW_CLOSE_DIFF':raw_data3['BTC_LOW_CLOSE_DIFF'],
    'DXY_CLOSE':raw_data3['DXY_CLOSE'],
    'DXY_HIGH_CLOSE_DIFF':raw_data3['DXY_HIGH_CLOSE_DIFF'],
    'DXY_LOW_CLOSE_DIFF':raw_data3['DXY_LOW_CLOSE_DIFF'],
    'TOTAL2_CLOSE':raw_data3['TOTAL2_CLOSE'],
    'TOTAL2_HIGH_CLOSE_DIFF':raw_data3['TOTAL2_HIGH_CLOSE_DIFF'],
    'TOTAL2_LOW_CLOSE_DIFF':raw_data3['TOTAL2_LOW_CLOSE_DIFF'],
    'TOTAL3_CLOSE':raw_data3['TOTAL3_CLOSE'],
    'TOTAL3_HIGH_CLOSE_DIFF':raw_data3['TOTAL3_HIGH_CLOSE_DIFF'],
    'TOTAL3_LOW_CLOSE_DIFF':raw_data3['TOTAL3_LOW_CLOSE_DIFF'],
    'MEAN9_REV_TOTAL_CLOSE':raw_data3['MEAN9_REV_TOTAL_CLOSE'],
    'MEAN20_REV_TOTAL_CLOSE':raw_data3['MEAN20_REV_TOTAL_CLOSE'],
    'MEAN50_REV_TOTAL_CLOSE':raw_data3['MEAN50_REV_TOTAL_CLOSE'],
    'MEAN100_REV_TOTAL_CLOSE':raw_data3['MEAN100_REV_TOTAL_CLOSE'],
    'MEAN9_REV_BTC_CLOSE':raw_data3['MEAN9_REV_BTC_CLOSE'],
    'MEAN20_REV_BTC_CLOSE':raw_data3['MEAN20_REV_BTC_CLOSE'],
    'MEAN50_REV_BTC_CLOSE':raw_data3['MEAN50_REV_BTC_CLOSE'],
    'MEAN100_REV_BTC_CLOSE':raw_data3['MEAN100_REV_BTC_CLOSE'],
    'MEAN9_REV_TOTAL2_CLOSE':raw_data3['MEAN9_REV_TOTAL2_CLOSE'],
    'MEAN20_REV_TOTAL2_CLOSE':raw_data3['MEAN20_REV_TOTAL2_CLOSE'],
    'MEAN50_REV_TOTAL2_CLOSE':raw_data3['MEAN50_REV_TOTAL2_CLOSE'],
    'MEAN100_REV_TOTAL2_CLOSE':raw_data3['MEAN100_REV_TOTAL2_CLOSE'],
    'MEAN9_REV_TOTAL3_CLOSE':raw_data3['MEAN9_REV_TOTAL3_CLOSE'],
    'MEAN20_REV_TOTAL3_CLOSE':raw_data3['MEAN20_REV_TOTAL3_CLOSE'],
    'MEAN50_REV_TOTAL3_CLOSE':raw_data3['MEAN50_REV_TOTAL3_CLOSE'],
    'MEAN100_REV_TOTAL3_CLOSE':raw_data3['MEAN100_REV_TOTAL3_CLOSE'],
    'MEAN9_REV_XRPUSD_CLOSE':raw_data3['MEAN9_REV_XRPUSD_CLOSE'],
    'MEAN20_REV_XRPUSD_CLOSE':raw_data3['MEAN20_REV_XRPUSD_CLOSE'],
    'MEAN50_REV_XRPUSD_CLOSE':raw_data3['MEAN50_REV_XRPUSD_CLOSE'],
    'MEAN100_REV_XRPUSD_CLOSE':raw_data3['MEAN100_REV_XRPUSD_CLOSE'],
    'BTC_DOM':raw_data3['BTC_DOM']
})


variables = ['BTC_CLOSE','DXY_CLOSE','TOTAL3_CLOSE','TOTAL_CLOSE']
target = 'BTC_CLOSE'
maxlag = 5000

from statsmodels.tsa.stattools import grangercausalitytests

#for var in variables:
#    print(f'\nGranger Causality Test: {var} → {target}')
#    test_data = raw_data2[[target, var]].dropna()
#    result = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)

#    for lag in range(1, maxlag + 1):
#        p_val = result[lag][0]['ssr_chi2test'][1]
#        if p_val < 0.05:
#            print(f'  Significant at lag {lag} with p-value {p_val:.4f}')

# Plotting
plt.figure(figsize=(16, 6))

for col in variables:
    ccf_values = ccf(main_data[target], main_data[col])[:maxlag]
    plt.plot(range(maxlag), ccf_values, label=f'XRPUSD vs {col}')

# Plot styling
plt.title('Cross-Correlation with up_pct')
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

ccf_values2 = ccf(main_data[target], main_data['BTC_CLOSE'])[0:2000]
max_value = np.max(ccf_values2)
max_index = np.where(ccf_values2 == max_value)[0][0]
print("Index of max correlation BTC_CLOSE:", max_index)
print("Max Correlation BTC_CLOSE:", max_value)

ccf_values3 = ccf(main_data[target], main_data['DXY_CLOSE'])[0:2000]
max_value2 = np.max(ccf_values3)
max_index2 = np.where(ccf_values3 == max_value2)[0][0]
print("Index of max correlation DXY_CLOSE:", max_index2)
print("Max Correlation DXY_CLOSE:", max_value2)

ccf_values4 = ccf(main_data[target], main_data['TOTAL_CLOSE'])[0:2000]
max_value3 = np.max(ccf_values4)
max_index3 = np.where(ccf_values4 == max_value3)[0][0]
print("Index of max correlation TOTAL_CLOSE:", max_index3)
print("Max Correlation TOTAL_CLOSE:", max_value3)

ccf_values5 = ccf(main_data[target], main_data['TOTAL3_CLOSE'])[0:2000]
max_value4 = np.max(ccf_values5)
max_index4 = np.where(ccf_values5 == max_value4)[0][0]
print("Index of max correlation TOTAL3_CLOSE:", max_index4)
print("Max Correlation TOTAL3_CLOSE:", max_value4)


