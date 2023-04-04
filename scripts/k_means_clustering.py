import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_in_dataframes():
    with open('saved_files/dfs.pkl', 'rb') as f:
        dfs = pd.read_pickle(f)
    with open('saved_files/dfs_mm.pkl', 'rb') as f:
        dfs_mm = pd.read_pickle(f)
    with open('saved_files/dfs_std.pkl', 'rb') as f:
        dfs_std = pd.read_pickle(f)
    return dfs, dfs_mm, dfs_std

def generate_elbow_plot_full_dim(dfs_mm):
    #define the range of k values to test
    k_range = range(1, 26)
    #store the mean distances
    mean_distances = []
    #calculate mean distances
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(dfs_mm['all'])
        distances = np.min(kmeans.transform(dfs_mm['all']), axis=1)
        mean_distances.append(np.mean(distances))

    # Plot the mean distances for each value of k
    plt.plot(k_range, mean_distances, 'o-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Mean distance from centroid')
    plt.title('Elbow plot for KMeans clustering')
    plt.savefig('figures/kmeans_elbow_100_clusters_18_dim.png')

def pick_cluster_representatives(dfs_mm,k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(dfs_mm['all'])
    representatives = kmeans.cluster_centers_
    return representatives

if __name__ == '__main__':
    dfs, dfs_mm, dfs_std = load_in_dataframes()
    generate_elbow_plot_full_dim(dfs_mm)
    centroids = pick_cluster_representatives(dfs_mm, 100)
    print(centroids)
    