#source : https://blog.patricktriest.com/analyzing-cryptocurrencies-python/
#done here till and not including 3.0

import os
import numpy as np
import pandas as pd
import pickle
import quandl
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.tree import export_graphviz

#break

import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
# py.init_notebook_mode(connected=True)
import matplotlib.pyplot as plt

#break

Epsilon = 45

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

#create classes

classes = np.zeros((btc_usd_price_kraken['Volume (Currency)'].size, 1))

print (classes.size)

for i in range (1, classes.size):
    if ( btc_usd_price_kraken['Open'][i] - btc_usd_price_kraken['Open'][i-1] ) > Epsilon :
        classes[i] = 2
    elif (( btc_usd_price_kraken['Open'][i] - btc_usd_price_kraken['Open'][i-1] ) < Epsilon) and ( btc_usd_price_kraken['Open'][i] > btc_usd_price_kraken['Open'][i-1] ) :
        classes[i] = 1
    else :
        classes[i] = 0

#finish classes

plt.subplot(211)

plt.plot( btc_usd_price_kraken['Open'] )
#indices =  np.arange(btc_usd_price_kraken['Volume (Currency)'].size)
#plt.plot( btc_usd_price_kraken['Volume (Currency)'], indices)
plt.xlabel('time')
plt.ylabel( 'Opening value of Bitcoin' )
#plt.show()
'''
#km = KMeans(n_clusters = 8)
agglom = AgglomerativeClustering( n_clusters = 6 )

prediction = agglom.fit_predict(btc_usd_price_kraken['Open'].values.reshape(-1,1))

#break

#prediction = agglom.predict(btc_usd_price_kraken['High'].values.reshape(-1,1))

indexer = np.arange(prediction.size)

plt.subplot(212)

plt.scatter(btc_usd_price_kraken['Open'], prediction)
#plt.plot(btc_usd_price_kraken['High'], prediction)
plt.xlabel("Opening value of Bitcoin")
plt.ylabel("Clusters")

plt.show()
'''

#do classification

LRG = LogisticRegression()

LRG.fit(btc_usd_price_kraken['Open'].values.reshape(-1,1), classes)

score = LRG.score(btc_usd_price_kraken['Open'].values.reshape(-1,1), classes)

predict = LRG.predict(btc_usd_price_kraken['Open'].values.reshape(-1,1))

#export_graphviz(DTC, out_file = "Decision_tree.dot")

plt.subplot(212)

plt.scatter(btc_usd_price_kraken['Open'], predict, color = 'blue')
#plt.scatter(btc_usd_price_kraken['Open'], classes, color = 'red')

plt.xlabel("Opening Value of Bitcoin")
plt.ylabel("Class")

plt.show()

errors = np.zeros((1,1))

for i in range (classes.size):
    if predict[i] != classes[i]:
        errors = np.append(errors, i)
        #print("predicted : ", predict[i], " but class is : ", classes[i])

print(errors.size)

print(score)
