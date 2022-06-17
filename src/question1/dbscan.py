from utils import read
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
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
    solar.loc[idx, 'Datetime'] = int(row['Datetime'].timetuple().tm_yday)
print(solar)
days = solar['Datetime'].to_numpy()
production = solar['Solar'].to_numpy()

nn = NearestNeighbors(n_neighbors=4)
neighbors = nn.fit(solar.to_numpy())

distances, indices = neighbors.kneighbors(solar.to_numpy())
distances = np.sort(distances[:, 3], axis=0)


i = np.arange(len(distances))
knee = KneeLocator(i, distances, S=1, curve='convex')

fig = plt.figure(figsize=(5, 5))
knee.plot_knee()
plt.xlabel("Points")
plt.ylabel("Distance")
plt.show()
print(distances[knee.knee])
dbscan_cluster1 = DBSCAN(eps=distances[knee.knee], min_samples=4)
dbscan_cluster1.fit(solar.to_numpy())

plt.scatter(days, production, c=dbscan_cluster1.labels_)
plt.xlabel("Days")
plt.ylabel("Production")

# sns.scatterplot(x='Datetime', y='value', data=solar.melt(['Datetime']))
plt.show()