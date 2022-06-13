from utils import read
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    sources = pd.read_pickle('dataset' + os.path.sep + 'pickleJar' + os.path.sep + 'sources.pkl')
    demand = pd.read_pickle('dataset' + os.path.sep + 'pickleJar' + os.path.sep + 'demand.pkl')
    print('Pickles retrieved')
except:
    sources = read("sources")
    demand = read("demand")
    sources.to_pickle('dataset' + os.path.sep + 'pickleJar' + os.path.sep + 'sources.pkl')
    demand.to_pickle('dataset' + os.path.sep + 'pickleJar' + os.path.sep + 'demand.pkl')
    print('Data pickled')
print(sources.describe())
print(sources.shape)
for name, values in sources.iteritems():
    sources[name].fillna(round(sources[name].median(skipna = True)), inplace = True)
print(sources.describe())
sources = sources.resample('1d').mean()
sources['Datetime'] = sources.index
solar = sources[['Datetime', 'Solar']]
for idx, row in solar.iterrows():
    solar.loc[idx, 'Datetime'] = row['Datetime'].timetuple().tm_yday
print(solar)
sns.scatterplot(x='Datetime', y='value', data=solar.melt(['Datetime']))
plt.show()