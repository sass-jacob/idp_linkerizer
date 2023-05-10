# idp_linkerizer
 Predicts an intrinsically disordered protein linker identity for smallest RMSD of amyloid beta 42 polymerized together with hexapeptide linkers versus the experimentally-determined 2beg.pdb 

 **(1)** Run reduced_linker_set.py using slurm_reduced_linker.sh to obtain the encodings for all possible hexapeptide linkers **excluding W, C, and P**.  
 **(2)** Run k_means_clustering.py to obtain a set of 100 encoded linkers that are clustered by taking the closest encoding to the centroids (KMedoids too memory intensive).  
 **(3)** Can run random_sampling.py to obtain a random set of unencoded linkers.  

**NOTE** generate_fasta_files.py can be used generally to generate fasta files relevant, but was not used in this implementation using ColabFold  

**(4)** Run average_2beg_workup.py to average the reference model across the 10 models provided experimentally.  
**(5)** Run AlphaFold2_Github_Submission.ipynb in a Google Colaboratory to run the prediction of structures.  
**(6)** Run compute_rmsd.ipynb in a Google Colaboratory to calculate the RMSDs between the predicted structures and the avg_2beg.pdb.  
**(7)** Feed rmsd values and pick acquisition function to run active_learning_loop.py and obtain new predictions to loop **5, 6, and 7**.  

Data analysis is performed using relevant files provided.

The linkers selected at each active learning loops and their acquisition function values are in the active_learning_results_dict directory. The linkers and their calculated rmsd values are in the rmsd directory. 
