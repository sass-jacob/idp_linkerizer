import numpy as np
import itertools

def generate_unencoded_linkers():
    #generate data set to perform clustering upon

    #array of amino acids to consider
    amino_acids = np.array(['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'])

    #linker length we want to generate (can be more than a single value - began at 6)
    min_link_len = 6
    max_link_len = 6

    #initialize blank linker array
    linkers = []

    #algorithm that generates all possible linkers of length 6 with given 'amino acid' array
    for link_size in range(min_link_len, max_link_len+1):
        for item in itertools.product(amino_acids, repeat=link_size):
            linkers.append(''.join(list(item)))
        
    return linkers

def save_linkerfile(linkers,filename):
    #save linker dataset as filename 
    np.savez_compressed(filename+'.npz', linkers)

#generate encodings for the linker set to represent the amino acids
#each amino acid represented by its charge, HLB (hydrophobicity), and MW (molecular weight)

#encode each amino acid
def encode_aa(aa):
    #map charge
    amino_acid_charges = {
        'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
        'E': -1, 'Q': 0, 'G': 0, 'H': 0, 'I': 0,
        'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
        'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
    }
    #map HLB
    hlb_scale = {
        'A': 8.1, 'C': 5.5, 'D': 13.0, 'E': 12.0, 'F': 5.2,
        'G': 9.0, 'H': 10.4, 'I': 4.9, 'K': 11.3, 'L': 4.9,
        'M': 5.7, 'N': 11.6, 'P': 8.0, 'Q': 10.5, 'R': 10.5,
        'S': 11.2, 'T': 9.1, 'V': 5.6, 'W': 4.4, 'Y': 6.2
    }
    
    #map MW
    amino_acid_weights = {
        'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.16,
        'E': 147.13, 'Q': 146.15, 'G': 75.07, 'H': 155.16, 'I': 131.18,
        'L': 131.18, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
        'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15
    }
    
    return amino_acid_charges[aa], round(hlb_scale[aa], 3), round(amino_acid_weights[aa], 3)

encode_aa('V')