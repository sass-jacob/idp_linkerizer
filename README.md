# idp_linkerizer
 Predicts linker identity for best distance map compared to experimental data of amyloid beta 42 peptides

 Begin with embedding each amino acid within a linker as a vector of their:  
 **(1)** Charge.  
 **(2)** Hydrophilic-Lypophilic Balance (HLB) index as a measure of hydrophobicity.  
 **(3)** Molecular Weight of Amino Acid Residue (Da).     

 MinMax and Standard scaling were performed on the entire linker set to compare effect on Principal Component Analysis results. Below shows the percentage of explained variance by each principal component graphed versus # of principal components for the linker set of all length 6 amino acid linkers.

 ![Explained Variance for Principal Component Analysis](https://github.com/sass-jacob/idp_linkerizer/blob/main/figures/PCA_explained_variance.png)

The projection of the first two principal components within the same plot is shown for all generated linkers:

![Principal components projections](https://github.com/sass-jacob/idp_linkerizer/blob/main/figures/pc_projections.png)

