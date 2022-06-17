from utils import read, source_names
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

THREADS = os.cpu_count()

print("Go, go, go!")
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

# Proprocessing
for name, values in sources.iteritems():
    sources[name].fillna(round(sources[name].median(skipna = True)), inplace = True)
print("Preprocessing complete")

sources = sources.resample('1d').mean()
sources['Datetime'] = sources.index
print(sources)

# Plot a graph of production for each type of power source
for source in source_names:
    df = sources[['Datetime', source]].copy()
    df['DayofYear'] = np.nan
    # Convert datetime format to day of year (1-365)
    for idx, row in df.iterrows():
        df.loc[idx, 'DayofYear'] = int(row['Datetime'].timetuple().tm_yday)
    data = df.to_numpy()
    min_pts = 4 #Minimum items in a cluster (Subject to change)
    print(min_pts)
    # Find optimal epsilon value
    nn = NearestNeighbors(n_neighbors=min_pts)
    neighbors = nn.fit(data[:, 2:0:-1])
    distances, indices = neighbors.kneighbors(data[:, 2:0:-1])
    distances = np.sort(distances[:, min_pts-1], axis=0)
    i = np.arange(len(distances))
    knee = KneeLocator(i, distances, S=1, curve='convex')
    epsilon = distances[knee.knee]
    # Apply the DBSCAN algorithm on the dataset
    dbscan_cluster = DBSCAN(eps=epsilon, min_samples=min_pts, n_jobs=THREADS)
    dbscan_cluster.fit(data[:, 2:0:-1])
    # Plot the Graph
    plt.figure(num=source, figsize=(5, 5))
    scatter = plt.scatter(data[:, 2], data[:, 1], c=dbscan_cluster.labels_, s=10, cmap='Set1')
    labels = [-1]
    for i in range(len(dbscan_cluster.labels_) - 1):
        labels.append(i)
    plt.legend(handles=scatter.legend_elements()[0], title= 'Cluster', labels=labels)
    plt.xlabel("Days")
    plt.ylabel(source)
    # Print outliers
    labeled_data = np.append(data, np.transpose([dbscan_cluster.labels_]), axis=1)
    output = pd.DataFrame(data=labeled_data, columns=['Datetime', 'Value', 'DayofYear', 'Label'])
    print(source + " outliers: (Total " + str(list(dbscan_cluster.labels_).count(-1)) + ")")
    for idx, row in output.iterrows():
        if row['Label'] == -1:
            print(row['Datetime'].strftime('%Y-%m-%d'))
    print(source + " Complete")

# Show all Graphs
plt.show()
