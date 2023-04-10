# idp_linkerizer
 Predicts an intrinsically disordered protein linker identity for best distance map compared to experimental data of amyloid beta 42 peptides

 Begin with embedding each amino acid within a linker as a vector of their:  
 **(1)** Charge.  
 **(2)** Hydrophilic-Lypophilic Balance (HLB) index as a measure of hydrophobicity.  
 **(3)** Molecular Weight of Amino Acid Residue (Da).     

 For example, the embedding of linker 'TMS', the resultant vector would be [0, 9.1, 119.12, 0, 5.7, 149.21, 0, 11.2, 105.09].  
 [Charge, HLB, MW, ..., Charge, HLB, MW].  
 (note that this linker is defined as only length 3)  
 
The total space for linker identities of length 6 containing all 20 amino acids spans 20^6 possible linkers.  
We removed a subset of amino acids based on similarity and avoidance of likely interactive moieties (such as Cysteine) to reduce this large space for clustering.  
This results in a subset of 13 amino acids shown below, which include positively charged, negatively charge, polar, and nonpolar amino acids.  
amino acid set {A, R, N, D, G, I, K, M, F, S, T, Y, V}

 MinMax and Standard scaling were performed on the embedded reduced amino acid linker set to compare effect on explained variance in Principal Component Analysis. Below shows the percentage of explained variance by each principal component graphed versus # of principal components for the linker set of all length 6 amino acid linkers.

 ![Explained Variance for Principal Component Analysis](https://github.com/sass-jacob/idp_linkerizer/blob/main/figures/reduced_explained_variance_plot.png)

The projection of the first two principal components the reduced linker set for visualization of data:

![Principal components projections](https://github.com/sass-jacob/idp_linkerizer/blob/main/figures/reduced_PCA_projection.png)



