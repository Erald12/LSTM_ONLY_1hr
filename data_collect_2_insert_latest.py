import pandas as pd
from tvDatafeed import TvDatafeed, Interval


# Load the CSV file into a DataFrame
main_data_history = pd.read_csv('main_data.csv',index_col='datetime')

def get_tv_data(symbol, exchange, interval, n_bars):
    tv = TvDatafeed()
    data = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)
    return data

main_interval = Interval.in_1_hour
number_of_bars = 8745

# Fetch data for multiple assets
symbol1 = "TOTAL"
exchange1 = "CRYPTOCAP"
interval1 = main_interval
n_bars1 = number_of_bars

symbol2 = "BTC"
exchange2 = "CRYPTOCAP"
interval2 = main_interval
n_bars2 = number_of_bars

symbol3 = "DXY"
exchange3 = "TVC"
interval3 = main_interval
n_bars3 = number_of_bars

symbol4 = 'XRP'
exchange4 = "CRYPTOCAP"
interval4 = main_interval
n_bars4 = number_of_bars

symbol5 = 'XRPUSD'
exchange5 = "CRYPTO"
interval5 = main_interval
n_bars5 = number_of_bars

# Get data for each symbol
data1 = get_tv_data(symbol1, exchange1, interval1, n_bars1)  # For TOTAL
data2 = get_tv_data(symbol2, exchange2, interval2, n_bars2)  # For BTC
data3 = get_tv_data(symbol3, exchange3, interval3, n_bars3)  # For DXY (Dollar Index)
data4 = get_tv_data(symbol4, exchange4, interval4, n_bars4)  # For ETH
data5 = get_tv_data(symbol5, exchange5, interval5, n_bars5)  # For ETHUSDT pair

# Align dataframes by their index
data1, data2 = data1.align(data2, join='inner')
_, data3 = data2.align(data3, join='outer')
data3 = data3.fillna(method='ffill')  # Forward fill missing values
_, data3 = data2.align(data3, join='inner')

_, data4 = data2.align(data4, join='inner')
_, data5 = data2.align(data5, join='inner')

# Verify if the indices are the same
same_index = data1.index.equals(data2.index)
same_index2 = data2.index.equals(data3.index)
same_index3 = data1.index.equals(data3.index)
same_index4 = data1.index.equals(data4.index)
same_index5 = data1.index.equals(data5.index)

print(same_index)  # True if they are the same, False otherwise
print(same_index2)  # True if they are the same, False otherwise
print(same_index3)  # True if they are the same, False otherwise
print(same_index4)  # True if they are the same, False otherwise
print(same_index5)  # True if they are the same, False otherwise

#modify data1
data1['high_close_diff'] = data1['high']-data1['close']
data1['low_close_diff'] = data1['low']-data1['close']

#modify data2
data2['high_close_diff'] = data2['high']-data2['close']
data2['low_close_diff'] = data2['low']-data2['close']

#modify data3
data3['high_close_diff'] = data3['high']-data3['close']
data3['low_close_diff'] = data3['low']-data3['close']

#modify data4
data4['high_close_diff'] = data4['high']-data4['close']
data4['low_close_diff'] = data4['low']-data4['close']

#modify data5
data5['high_close_diff'] = data5['high']-data5['close']
data5['low_close_diff'] = data5['low']-data5['close']

# Combine relevant features into a single dataframe for the model
main_data = pd.DataFrame({
    'TOTAL_MC_CLOSE':data1['close'],
    'TOTAL_MC_OPEN':data1['open'],
    'TOTAL_MC_HIGH':data1['high'],
    'TOTAL_MC_LOW':data1['low'],
    'TOTAL_MC_HIGH_CLOSE_DIFF': data1['high_close_diff'],
    'TOTAL_MC_LOW_CLOSE_DIFF':data1['low_close_diff'],
    'BTC_MC_CLOSE':data2['close'],
    'BTC_MC_OPEN':data2['open'],
    'BTC_MC_HIGH':data2['high'],
    'BTC_MC_LOW':data2['low'],
    'BTC_MC_HIGH_CLOSE_DIFF': data2['high_close_diff'],
    'BTC_MC_LOW_CLOSE_DIFF':data2['low_close_diff'],
    'DXY_CLOSE':data3['close'],
    'DXY_OPEN':data3['open'],
    'DXY_HIGH':data3['high'],
    'DXY_LOW':data3['low'],
    'DXY_HIGH_CLOSE_DIFF': data3['high_close_diff'],
    'DXY_LOW_CLOSE_DIFF':data3['low_close_diff'],
    'XRP_MC_CLOSE':data4['close'],
    'XRP_MC_OPEN':data4['open'],
    'XRP_MC_HIGH':data4['high'],
    'XRP_MC_LOW':data4['low'],
    'XRP_MC_HIGH_CLOSE_DIFF': data4['high_close_diff'],
    'XRP_MC_LOW_CLOSE_DIFF':data4['low_close_diff'],
    'XRPUSD_CLOSE':data5['close'],
    'XRPUSD_OPEN':data5['open'],
    'XRPUSD_HIGH':data5['high'],
    'XRPUSD_LOW':data5['low'],
    'XRPUSD_HIGH_CLOSE_DIFF': data5['high_close_diff'],
    'XRPUSD_LOW_CLOSE_DIFF':data5['low_close_diff'],
    'TOTAL2_CLOSE':data1['close']-data2['close'],
    'TOTAL2_OPEN':data1['open']-data2['open'],
    'TOTAL2_HIGH':data1['high']-data2['high'],
    'TOTAL2_LOW':data1['low']-data2['low'],
    'TOTAL2_HIGH_CLOSE_DIFF':(data1['high']-data2['high'])-(data1['close']-data2['close']),
    'TOTAL2_LOW_CLOSE_DIFF':(data1['low']-data2['low'])-(data1['close']-data2['close']),
    'TOTAL3_CLOSE':(data1['close']-data2['close'])-data4['close'],
    'TOTAL3_OPEN':(data1['open']-data2['open'])-data4['open'],
    'TOTAL3_HIGH':(data1['high']-data2['high'])-data4['high'],
    'TOTAL3_LOW':(data1['low']-data2['low'])-data4['low'],
    'TOTAL3_HIGH_CLOSE_DIFF':((data1['high']-data2['high'])-(data1['close']-data2['close']))-data4['high_close_diff'],
    'TOTAL3_LOW_CLOSE_DIFF':((data1['low']-data2['low'])-(data1['close']-data2['close']))-data4['high_close_diff']
})


# **Convert index to datetime format to avoid TypeError**
main_data_history.index = pd.to_datetime(main_data_history.index)
main_data.index = pd.to_datetime(main_data.index)

# **Step 1: Remove old entries that exist in the new data**
main_data_history = main_data_history[~main_data_history.index.isin(main_data.index)]

# **Step 2: Merge old and new data**
updated_data = pd.concat([main_data_history, main_data])

# **Step 3: Sort by datetime index and remove duplicates**
updated_data = updated_data.sort_index().drop_duplicates()

#Drop the last row
new_data = updated_data.drop(updated_data.index[-1])
print(len(new_data['TOTAL_MC_CLOSE']))

# **Step 4: Save the updated dataset**
new_data.to_csv('main_data.csv', index=True)

print("Data successfully updated and saved based on index.")



