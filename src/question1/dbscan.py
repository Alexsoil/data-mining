import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
THREADS = os.cpu_count()

# Input dataframe with data to process, and a list of the column names
# Note! Input must not have NaN values! make sure to run preprocessing before
def outlier_detector(raw: pd.DataFrame, col_list: list) -> None:
    # Resample data so that each row represents the average values of a single day
    raw = raw.resample('1d').mean()
    # Insert the index (the date that the values represent)
    raw['Datetime'] = raw.index

    for item in col_list:
        df = raw[['Datetime', item]].copy()
        df['DayofYear'] = np.nan
        # Convert datetime format to day of year (1-365)
        for idx, row in df.iterrows():
            df.loc[idx, 'DayofYear'] = int(row['Datetime'].timetuple().tm_yday)
        data = df.to_numpy()
        min_pts = 4 # Minimum items in a cluster (Subject to change)
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
        plt.figure(num=item, figsize=(5, 5))
        scatter = plt.scatter(data[:, 2], data[:, 1], c=dbscan_cluster.labels_, s=10, cmap='gnuplot')
        labels = [-1]
        for i in range(len(dbscan_cluster.labels_) - 1):
            labels.append(i)
        plt.legend(handles=scatter.legend_elements()[0], title= 'Cluster', labels=labels)
        plt.xlabel("Days")
        plt.ylabel(item)
        plt.savefig('images' + os.path.sep + str(item))
        # Print outliers
        labeled_data = np.append(data, np.transpose([dbscan_cluster.labels_]), axis=1)
        output = pd.DataFrame(data=labeled_data, columns=['Datetime', 'Value', 'DayofYear', 'Label'])
        print(item + " outliers: (Total " + str(list(dbscan_cluster.labels_).count(-1)) + ")")
        for idx, row in output.iterrows():
            if row['Label'] == -1:
                print(row['Datetime'].strftime('%Y-%m-%d'))
        print(item + " Complete")

    # Show all Graphs
    plt.show()
