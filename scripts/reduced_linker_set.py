import numpy as np
import itertools
import pandas as pd

def generate_unencoded_linkers():
    #generate data set to perform clustering upon
    #removed W due to low likelihood of linker functionality
    #removed C to prevent disulfide bonding in linker
    #removed P due to low likelihood of linker functionality
    amino_acids = np.array(['A', 'R', 'N', 'D', 'G', 'I', 'K', 'M', 'F', 'S', 'T', 'Y', 'V', 'Q', 'E', 'L', 'H'])
    #linker length we want to generate (can be more than a single value - began at 6)
    min_link_len = 6
    max_link_len = 6
    #initialize blank linker array
    linkers = []
    #algorithm that generates all possible linkers of lengths min -> max with given 'amino acid' array
    for link_size in range(min_link_len, max_link_len+1):
        for item in itertools.product(amino_acids, repeat=link_size):
            linkers.append(''.join(list(item)))   
    return linkers

def save_linkerfile(linkers,filename):
    #save linker dataset as filename 
    np.savez_compressed(filename+'.npz', linkers)

def set_features(input_peptide):
    out_row = []
    peptide = [*input_peptide]
    for letter in peptide:
        which_amino_index = np.where(np.array(amino_acids) == letter)[0]
        # pick the row corresponding to the amino acid to be appended to the output
        feature = np.array(amino_acid_df[['VHSE1', 'VHSE2', 'VHSE3', 'VHSE4','VHSE5', 'VHSE6', 'VHSE7', 'VHSE8']].to_numpy()[which_amino_index])[0]
        out_row = np.append(out_row,feature)
    return out_row

def load_in_features():
    amino_acid_df = pd.read_csv("../saved_files/amino_acid.csv")
    amino_acid_df = amino_acid_df.set_index('Amino Acids')
    return amino_acid_df

if __name__ == '__main__':
    #reduced_linkers = generate_unencoded_linkers()
    reduced_linkers = np.load('../saved_files/reduced_linkers.npz')['arr_0']
    amino_acid_df = load_in_features()
    featurized = np.zeros((len(reduced_linkers), 6 * len(np.array(amino_acid_df[['VHSE1', 'VHSE2', 'VHSE3', 'VHSE4','VHSE5', 'VHSE6', 'VHSE7', 'VHSE8']])[0])))    
    amino_acids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    # featurize all of the data
    index = 0
    for linker in reduced_linkers:
        #print(set_features(linker))
        featurized[index] = set_features(linker)
        #print(np.shape(featurized))
        if index % 100000 == 0:
            print((index, featurized[index]))
        index += 1

    print(featurized[-10:])
    #save_linkerfile(featurized,'../saved_files/encoded_reduced_linkers')