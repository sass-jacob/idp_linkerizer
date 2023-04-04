import pandas as pd
import numpy as np
from principal_component_analysis import load_in_data
from principal_component_analysis import make_pandas_dataframes
from principal_component_analysis import compute_PCA_embeddings
import umap.umap_ as umap
import seaborn as sns
import matplotlib.pyplot as plt


def compute_umap_embeddings(dfs, df_noscaler, dfs_mm, df_mmscaler, dfs_std, df_stdscaler):
    # 2-dim umap embedding
    nneigh = int(np.sqrt(df_noscaler.shape[1]))
    print(nneigh)
#    dfs['umap2'] = pd.DataFrame(umap.UMAP(n_components=2,
#                    n_neighbors=nneigh).fit_transform(df_noscaler),
#                    index=df_noscaler.index,
#                    columns=["UMAP1", "UMAP2"])
#    print("1 done")
    dfs_mm['umap2'] = pd.DataFrame(umap.UMAP(n_components=2,
                    n_neighbors=nneigh).fit_transform(df_mmscaler),
                    index=df_mmscaler.index,
                    columns=["UMAP1", "UMAP2"])
    print("2 done")
    dfs_std['umap2'] = pd.DataFrame(umap.UMAP(n_components=2,
                    n_neighbors=nneigh).fit_transform(df_stdscaler),
                    index=df_stdscaler.index,
                    columns=["UMAP1", "UMAP2"])
    return dfs, dfs_std, dfs_mm

def plot_umap_embeddings(dfs, dfs_mm, dfs_std):
    f2, ax2 = plt.subplots(1, 3, figsize=(15, 5))
    sns.scatterplot('UMAP1', 'UMAP2', data=dfs['umap2'], alpha=0.5, ax=ax2[0]).set_title("UMAP projection")
    sns.scatterplot('UMAP1', 'UMAP2', data=dfs_mm['umap2'], alpha=0.5, ax=ax2[1]).set_title("UMAP projection with MinMax")
    sns.scatterplot('UMAP1', 'UMAP2', data=dfs_std['umap2'], alpha=0.5, ax=ax2[2]).set_title("UMAP projection with Standard")
    plt.savefig('figures/reduced_umap_projection.png')    
    return

if __name__ == '__main__':
    encoded_linkers = load_in_data() #currently loading in reduced dataset
    df_noscaler, df_stdscaler, df_mmscaler = make_pandas_dataframes(encoded_linkers)
    dfs, dfs_mm, dfs_std = compute_PCA_embeddings(df_noscaler, df_stdscaler, df_mmscaler)
    dfs, dfs_mm, dfs_std = compute_umap_embeddings(dfs, df_noscaler, dfs_std, df_stdscaler, dfs_mm, df_mmscaler)
    plot_umap_embeddings(dfs, dfs_mm, dfs_std)
