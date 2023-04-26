import numpy as np
import itertools

print('beginning')

def generate_unencoded_linkers():
    #generate data set to perform clustering upon

    #array of amino acids to consider
    #removed W due to low likelihood of linker functionality
    #removed C to prevent disulfide bonding in linker
    #removed P due to low likelihood of linker functionality
    amino_acids = np.array(['A', 'R', 'N', 'D', 'G', 'I', 'K', 'M', 'F', 'S', 'T', 'Y', 'V', 'Q', 'E', 'L', 'H'])

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

#reduced_linkers = generate_unencoded_linkers()
reduced_linkers = np.load('../saved_files/reduced_linkers.npz')['arr_0']
print(np.shape(reduced_linkers))
#save_linkerfile(reduced_linkers,'../saved_files/reduced_linkers')

import pandas as pd
amino_acid_df = pd.read_csv("../saved_files/amino_acid.csv")
amino_acid_df = amino_acid_df.set_index('Amino Acids')
featurized = np.zeros((len(reduced_linkers), 6 * len(np.array(amino_acid_df[['VHSE1', 'VHSE2', 'VHSE3', 'VHSE4','VHSE5', 'VHSE6', 'VHSE7', 'VHSE8']])[0])))
print(np.shape(featurized))
amino_acids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

def set_features(input_peptide):
    out_row = []
    peptide = [*input_peptide]
    for letter in peptide:
        which_amino_index = np.where(np.array(amino_acids) == letter)[0]
        # pick the row corresponding to the amino acid to be appended to the output
        #print(amino_acid_df[['VHSE1', 'VHSE2', 'VHSE3', 'VHSE4','VHSE5', 'VHSE6', 'VHSE7', 'VHSE8']].to_numpy()[which_amino_index])
        feature = np.array(amino_acid_df[['VHSE1', 'VHSE2', 'VHSE3', 'VHSE4','VHSE5', 'VHSE6', 'VHSE7', 'VHSE8']].to_numpy())[which_amino_index]
        out_row = np.append(out_row,feature)
    return out_row

# featurize all of the data
index = 0
for linker in reduced_linkers:
    featurized[index, :] = set_features(linker)
    index += 1
    if index % 10000 == 0:
        print(index)

save_linkerfile(featurized,'../saved_files/encoded_reduced_linkers')
