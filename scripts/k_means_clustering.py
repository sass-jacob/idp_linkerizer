import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import sklearn

#load in the encoded linkerset
def load_in_encoded_linkers():
    encoded_linkers = np.load('../saved_files/encoded_reduced_linkers.npz')['arr_0']
    return encoded_linkers

#elbow plot (unused)
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

#clustering using kmeans clustering
def pick_cluster_representatives(encoded_linkers,k):
    #fit kmeans
    kmeans = KMeans(n_clusters=k, n_init="auto", max_iter=5000, random_state=42)
    kmeans.fit(encoded_linkers)
    print('fit kmeans')
    distances = kmeans.transform(encoded_linkers)
    print('distances shape: {}'.format(np.shape(distances)))
    closest_indices = np.array(np.argmin(distances, axis=0))
    print('indices shape: {}'.format(np.shape(closest_indices))) 
    representatives = np.zeros((k, 48))
    #find the representatives
    for i in range(k):
        representatives[i,:] = encoded_linkers[closest_indices[i]]
    return representatives, closest_indices


def save_nparray(nparray,filename):
    #save linker dataset as filename 
    np.savez_compressed(filename+'.npz', nparray)
    return

#decode the encoded representatives
def decode(representatives, aa_df):
    rep_linkers = []
    #go through each representative
    for representative in representatives:
        sequence = ''
        for encoding_index in range(len(representative)):
            #every 8th index is a VHSE1 encoding
            if (encoding_index - 1) % 8 == 0:
                sequence += (aa_df.loc[aa_df['VHSE1'] == representative[encoding_index-1]]).index
        rep_linkers.append(sequence[0])
    return rep_linkers

#too memory intensive because computing pairwise distances
def pick_cluster_medoids(encoded_linkers,k):
    #fit kmedoids
    kmedoids = KMedoids(n_clusters=k, metric='euclidean')
    kmedoids.fit(encoded_linkers)
    medoid_indices = kmedoids.medoid_indices_
    representatives = encoded_linkers[medoid_indices]
    return representatives, medoid_indices


if __name__ == '__main__':
    encoded_linkers = load_in_encoded_linkers()
    #medoids, medoid_indices = pick_cluster_medoids(encoded_linkers, 100) -> too memory intensive
    #centroids, closest_indices = pick_cluster_representatives(encoded_linkers, 100)
    centroids = np.load('../saved_files/encoded_near_centroid.npz')['arr_0']
    closest_indices = np.load('../saved_files/near_centroid_indices.npz')['arr_0']
    print(centroids)
    print(closest_indices)
    #save_nparray(centroids, '../saved_files/encoded_near_centroid')
    #save_nparray(closest_indices, '../saved_files/near_centroid_indices')
    amino_acid_df = pd.read_csv("../saved_files/amino_acid.csv")
    amino_acid_df = amino_acid_df.set_index('Amino Acids')
    print(np.shape(amino_acid_df))
    representatives = decode(centroids, amino_acid_df)
    print(np.shape(representatives))
    #save_nparray(representatives, '../saved_files/decoded_representatives')
