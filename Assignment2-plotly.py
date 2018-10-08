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
py.init_notebook_mode(connected=True)

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
data=go.Data([btc_trace])
layout=go.Layout(title="First Plot", xaxis={'title':'x1'}, yaxis={'title':'x2'})
figure=go.Figure(data=data,layout=layout)
#py.iplot([btc_trace], filename = "pyplot 1")
#py.iplot(figure, filename='pyguide_1')
py.offline.plot([btc_trace])

#break

# Pull pricing data for 3 more BTC exchanges
'''
'''
