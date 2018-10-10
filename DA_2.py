#source : https://blog.patricktriest.com/analyzing-cryptocurrencies-python/
#done here till and not including 3.0

import os
import numpy as np
import pandas as pd
import pickle
import quandl
from datetime import datetime

#break

import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
# py.init_notebook_mode(connected=True)
import matplotlib.pyplot as plt

#break

def get_quandl_data(quandl_id):
    '''Download and cache Quandl dataseries'''
    cache_path = '{}.pkl'.format(quandl_id).replace('/','-')
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)   
        print('Loaded {} from cache'.format(quandl_id))
    except (OSError, IOError) as e:
        print('Downloading {} from Quandl'.format(quandl_id))
        df = quandl.get(quandl_id, returns="pandas")
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(quandl_id, cache_path))
    return df
	
#break

# Pull Kraken BTC price exchange data
btc_usd_price_kraken = get_quandl_data('BCHARTS/KRAKENUSD')

btc_usd_price_kraken.head()

#break

# Chart the BTC pricing data
btc_trace = go.Scatter(x=btc_usd_price_kraken.index, y=btc_usd_price_kraken['Weighted Price'])
#data=go.Data([btc_trace])
#layout=go.Layout(title="First Plot", xaxis={'title':'x1'}, yaxis={'title':'x2'})
#figure=go.Figure(data=data,layout=layout)
#py.iplot([btc_trace], filename = "pyplot 1")
#py.iplot(figure, filename='pyguide_1')
#py.offline.plot([btc_trace])

plt.figure(1)

plt.plot( btc_usd_price_kraken.index, btc_usd_price_kraken['Volume (BTC)'] )
plt.xlabel('time')
plt.ylabel( 'Volume of Bitcoin' )
plt.show()

plt.figure(2)

plt.hist( btc_usd_price_kraken['Volume (Currency)'] )
#plt.xlabel('time')
plt.ylabel( 'Volume of Currency' )
plt.show()

plt.figure(3)

plt.plot( btc_usd_price_kraken.index, btc_usd_price_kraken['High'] )
plt.xlabel('time')
plt.ylabel( 'High Price in USD' )
plt.show()

plt.figure(4)

plt.hist( btc_usd_price_kraken['Low'] )
#plt.xlabel('time')
plt.ylabel( 'Low Price in USD' )
plt.show()

#break

# Pull pricing data for 3 more BTC exchanges

'''

exchanges = ['COINBASE','BITSTAMP','ITBIT']

exchange_data = {}

exchange_data['KRAKEN'] = btc_usd_price_kraken

for exchange in exchanges:
    exchange_code = 'BCHARTS/{}USD'.format(exchange)
    btc_exchange_df = get_quandl_data(exchange_code)
    exchange_data[exchange] = btc_exchange_df
	
def merge_dfs_on_column(dataframes, labels, col):
    # Merge a single column of each dataframe into a new combined dataframe
    series_dict = {}
    for index in range(len(dataframes)):
        series_dict[labels[index]] = dataframes[index][col]
        
    return pd.DataFrame(series_dict)
	
#break

# Merge the BTC price dataseries' into a single dataframe
btc_usd_datasets = merge_dfs_on_column(list(exchange_data.values()), list(exchange_data.keys()), 'Weighted Price')

btc_usd_datasets.tail()

#break

def df_scatter(df, title, seperate_y_axis=False, y_axis_label='', scale='linear', initial_hide=False):
    #Generate a scatter plot of the entire dataframe
    label_arr = list(df)
    series_arr = list(map(lambda col: df[col], label_arr))
    
    layout = go.Layout(
        title=title,
        legend=dict(orientation="h"),
        xaxis=dict(type='date'),
        yaxis=dict(
            title=y_axis_label,
            showticklabels= not seperate_y_axis,
            type=scale
        )
    )
    
    y_axis_config = dict(
        overlaying='y',
        showticklabels=False,
        type=scale )
    
    visibility = 'visible'
    if initial_hide:
        visibility = 'legendonly'
        
    # Form Trace For Each Series
    trace_arr = []
    for index, series in enumerate(series_arr):
        trace = go.Scatter(
            x=series.index, 
            y=series, 
            name=label_arr[index],
            visible=visibility
        )
        
        # Add seperate axis for the series
        if seperate_y_axis:
            trace['yaxis'] = 'y{}'.format(index + 1)
            layout['yaxis{}'.format(index + 1)] = y_axis_config    
        trace_arr.append(trace)

    fig = go.Figure(data=trace_arr, layout=layout)
    py.offline.plot(fig, filename = 'plot3')
	
# Plot all of the BTC exchange prices
df_scatter(btc_usd_datasets, 'Bitcoin Price (USD) By Exchange')

#break

# Remove "0" values
btc_usd_datasets.replace(0, np.nan, inplace=True)

# Plot the revised dataframe
#df_scatter(btc_usd_datasets, 'Bitcoin Price (USD) By Exchange')

#break

# Calculate the average BTC price as a new column
btc_usd_datasets['avg_btc_price_usd'] = btc_usd_datasets.mean(axis=1)

# Plot the average BTC price
btc_trace = go.Scatter(x=btc_usd_datasets.index, y=btc_usd_datasets['avg_btc_price_usd'])
py.offline.plot([btc_trace], filename = 'plot2')
'''
