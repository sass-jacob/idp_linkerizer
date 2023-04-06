import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn

def load_in_dataframes():
    #files will need to be referenced from local filesystem because .pkl files exceed the github limit
    #adjust file directory when necessary
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
    distances = kmeans.transform(dfs_mm['all'])
    closest_indices = np.array(np.argmin(distances, axis=0))
    representatives = np.zeros((k, 18))
    i = 0
    for index in closest_indices:
        representatives[i,:] = dfs_mm['all'].values[index]
        i += 1
    return representatives

def save_nparray(nparray,filename):
    #save linker dataset as filename 
    np.savez_compressed(filename+'.npz', nparray)
    return

def unstandardize_and_decode(representatives, std_array):
    print('representatives')
    print(representatives)
    mmscaler = sklearn.preprocessing.MinMaxScaler()
    scaled = mmscaler.fit_transform(std_array)
    print('scaled')
    print(scaled)
    unstd_reps = mmscaler.inverse_transform(representatives)
    print(unstd_reps)
    #map HLB
    hlb_scale = {
        'A': 8.1, 'C': 5.5, 'D': 13.0, 'E': 12.0, 'F': 5.2,
        'G': 9.0, 'H': 10.4, 'I': 4.9, 'K': 11.3, 'L': 4.9,
        'M': 5.7, 'N': 11.6, 'P': 8.0, 'Q': 10.5, 'R': 10.5,
        'S': 11.2, 'T': 9.1, 'V': 5.6, 'W': 4.4, 'Y': 6.2
    }
    hlb_scale_inv = {v: k for k, v in hlb_scale.items()}
    rep_linkers = []
    for unstd_rep in unstd_reps:
        sequence = ''
        for encoding_index in range(len(unstd_rep)):
            if (encoding_index - 1) % 3 == 0:
                sequence += hlb_scale_inv[unstd_rep[encoding_index]]
        rep_linkers.append(sequence)
    return rep_linkers

if __name__ == '__main__':
    dfs, dfs_mm, dfs_std = load_in_dataframes()
    #generate_elbow_plot_full_dim(dfs_mm)
    centroids = pick_cluster_representatives(dfs_mm, 100)
    save_nparray(centroids, 'saved_files/encoded_reps')
    #centroids = np.load('saved_files/encoded_reps.npz')['arr_0']
    representative_linkers = unstandardize_and_decode(centroids, dfs['all'])
    save_nparray(representative_linkers, 'saved_files/decoded_reps')
    print(representative_linkers)