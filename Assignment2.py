#source : https://blog.patricktriest.com/analyzing-cryptocurrencies-python/
#done here till and not including 3.0

import os
import numpy as np
import pandas as pd
import pickle
import quandl
from datetime import datetime

#break

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

plt.plot(btc_usd_price_kraken.index, btc_usd_price_kraken['Weighted Price'])
plt.xlabel('Year')
plt.ylabel('Price in USD')
plt.show()

#break

# Pull pricing data for 3 more BTC exchanges
'''
'''
