import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn

def load_in_encoded_linkers():
    encoded_linkers = np.load('../saved_files/encoded_reduced_linkers.npz')['arr_0']
    return encoded_linkers

def generate_elbow_plot_full_dim(encoded_linkers):
    #define the range of k values to test
    k_range = range(1, 26)
    #store the mean distances
    mean_distances = []
    #calculate mean distances
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(encoded_linkers)
        distances = np.min(kmeans.transform(encoded_linkers), axis=1)
        mean_distances.append(np.mean(distances))

    # Plot the mean distances for each value of k
    plt.plot(k_range, mean_distances, 'o-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Mean distance from centroid')
    plt.title('Elbow plot for KMeans clustering')
    plt.savefig('figures/kmeans_elbow.png')

def pick_cluster_representatives(encoded_linkers,k):
    kmeans = KMeans(n_clusters=k, n_init="auto")
    kmeans.fit(encoded_linkers)
    print('fit kmeans')
    distances = kmeans.transform(encoded_linkers)
    closest_indices = np.array(np.argmin(distances, axis=0))
    representatives = np.zeros((k, 48))
    #find the representatives
    i = 0
    for index in closest_indices:
        representatives[i,:] = encoded_linkers[index]
        i += 1
    return representatives, closest_indices

def save_nparray(nparray,filename):
    #save linker dataset as filename 
    np.savez_compressed(filename+'.npz', nparray)
    return

def decode(representatives, std_array):
    mmscaler = sklearn.preprocessing.MinMaxScaler()
    unstd_reps = mmscaler.inverse_transform(representatives)
    #map HLB
    rep_linkers = []
    for unstd_rep in unstd_reps:
        sequence = ''
        for encoding_index in range(len(unstd_rep)):
            if (encoding_index - 1) % 3 == 0:
                sequence += hlb_scale_inv[unstd_rep[encoding_index]]
        rep_linkers.append(sequence)
    return rep_linkers

if __name__ == '__main__':
    encoded_linkers = load_in_encoded_linkers()
    print(np.shape(encoded_linkers))
    #centroids, closest_indices = pick_cluster_representatives(encoded_linkers, 100)
    centroids = np.load('../saved_files/encoded_reps.npz')['arr_0']
    closest_indices = np.load('../saved_files/medoid_indices.npz')['arr_0']
    print(np.shape(centroids))
    print(np.shape(closest_indices))
    #save_nparray(centroids, '../saved_files/encoded_reps')
    #save_nparray(closest_indices, '../saved_files/medoid_indices')
    
    #representative_linkers = unstandardize_and_decode(centroids, dfs['all'])
    #save_nparray(representative_linkers, 'saved_files/decoded_reps')
