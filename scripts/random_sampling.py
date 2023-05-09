import numpy as np
import random

def save_linkerfile(linkers,filename):
    #save linker dataset as filename 
    np.savez_compressed(filename+'.npz', linkers)

if __name__ == '__main__':
    linkers = np.load('../saved_files/reduced_linkers.npz')['arr_0']
    print(np.shape(linkers))
    random_linkers = linkers[random.sample(range(len(linkers)), 100)]
    print(random_linkers)
    #save_linkerfile(random_linkers, '../saved_files/insert_random_linkers_name_here')
