#source : https://blog.patricktriest.com/analyzing-cryptocurrencies-python/
#done here till and not including 3.0

import os
import numpy as np
import pandas as pd
import pickle
import quandl
from datetime import datetime

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#break

'''import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff'''
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
    
    if ( btc_usd_price_kraken['Open'][i] > btc_usd_price_kraken['Open'][i-1] ) :
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

X_train, X_test, y_train, y_test = train_test_split( btc_usd_price_kraken['Open'].values.reshape(-1,1), classes)

#X - data, y - class

DTC = DecisionTreeClassifier()

DTC.fit(X_train, y_train)

score = DTC.score(X_test, y_test)

predict = DTC.predict(X_test)

export_graphviz(DTC, out_file = "Decision_tree.dot")

plt.subplot(212)

plt.scatter(btc_usd_price_kraken['Open'], predict, color = 'blue')
#plt.scatter(btc_usd_price_kraken['Open'], classes, color = 'red')

plt.xlabel("Opening Value of Bitcoin")
plt.ylabel("Class")

plt.show()

errors = np.zeros((1,1))

'''
for i in range (classes.size):
    if predict[i] != classes[i]:
        errors = np.append(errors, i)
        #print("predicted : ", predict[i], " but class is : ", classes[i])
'''

#print(errors.size)

print("score calculated from api call : " + score)

true_pos, true_neg = 0,0
false_pos, false_neg = 0,0

#positive -> 1, negative -> 0
'''
for i in range (predict.size):
	if predict[i] == y_test[i]:
		if predict[i] == 0:
			true_neg += 1
		else:
			true_pos += 1
	else:
		if predict[i] == 0:
			false_neg'''

conf_mat = confusion_matrix( y_test, predict )

tn, fp, fn, tp = conf_mat.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

precision = tp / ( tp + fp )
recall = sensitivity

f1_score = 2 * precision * recall / ( precision + recall )