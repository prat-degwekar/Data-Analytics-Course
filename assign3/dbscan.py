#source : https://blog.patricktriest.com/analyzing-cryptocurrencies-python/
#done here till and not including 3.0

import os
import numpy as np
import pandas as pd
import pickle
import quandl
from datetime import datetime

from sklearn.cluster import DBSCAN

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

#break

# Remove "0" values

btc_usd_datasets.replace(np.nan, 0, inplace=True)

# Plot the revised dataframe
#break

# Calculate the average BTC price as a new column
btc_usd_datasets['avg_btc_price_usd'] = btc_usd_datasets.mean(axis=1)

plt.subplot(211)

#plt.plot( btc_usd_datasets['avg_btc_price_usd'], btc_usd_datasets.index )
plt.plot( btc_usd_datasets['avg_btc_price_usd'] )
plt.xlabel('time')
plt.ylabel( 'Average Price of a Bitcoin in USD across 3 exchanges' )
#plt.show()


#km = KMeans(n_clusters = 8)
dbscan = DBSCAN(eps = 250)

prediction = dbscan.fit_predict(btc_usd_datasets['avg_btc_price_usd'].values.reshape (-1, 1))

#break

#prediction = agglom.predict(btc_usd_price_kraken['High'].values.reshape(-1,1))

indexer = np.arange(prediction.size)

plt.subplot(212)

plt.scatter(btc_usd_datasets['avg_btc_price_usd'], prediction)
#plt.plot(btc_usd_price_kraken['High'], prediction)
plt.xlabel("Average Bitcoin Price in USD across 3 exchanges")
plt.ylabel("Clusters")

plt.show()
