#source : https://blog.patricktriest.com/analyzing-cryptocurrencies-python/
#done here till and not including 3.0

import os
import numpy as np
import pandas as pd
import pickle
import quandl
from datetime import datetime

from sklearn.cluster import AgglomerativeClustering

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

plt.subplot(211)

plt.plot( btc_usd_price_kraken['Volume (Currency)'] )
#indices =  np.arange(btc_usd_price_kraken['Volume (Currency)'].size)
#plt.plot( btc_usd_price_kraken['Volume (Currency)'], indices)
plt.xlabel('time')
plt.ylabel( 'Volume of Currency Traded' )
#plt.show()

#km = KMeans(n_clusters = 8)
agglom = AgglomerativeClustering( n_clusters = 6 )

prediction = agglom.fit_predict(btc_usd_price_kraken['Volume (Currency)'].values.reshape(-1,1))

#break

#prediction = agglom.predict(btc_usd_price_kraken['High'].values.reshape(-1,1))

indexer = np.arange(prediction.size)

plt.subplot(212)

plt.scatter(btc_usd_price_kraken['Volume (Currency)'], prediction)
#plt.plot(btc_usd_price_kraken['High'], prediction)
plt.xlabel("Volume (Currency)")
plt.ylabel("Clusters")

plt.show()
