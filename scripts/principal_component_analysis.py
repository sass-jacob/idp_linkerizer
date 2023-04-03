import numpy as np
import sklearn
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def load_in_data():
    #load data from saved files if necessary
    encoded_linkers = np.load('saved_files/reduced_encoded_linkers.npz', allow_pickle=True)['arr_0']
    print(np.shape(encoded_linkers))
    return encoded_linkers

def make_pandas_dataframes(encoded_linkers):
    #make pandas dataframes
    mmscaler = sklearn.preprocessing.MinMaxScaler()
    stdscaler = sklearn.preprocessing.StandardScaler()
    df_stdscaler = pd.DataFrame(stdscaler.fit_transform(encoded_linkers))
    df_mmscaler = pd.DataFrame(mmscaler.fit_transform(encoded_linkers))
    df_noscaler = pd.DataFrame(encoded_linkers)
    return df_noscaler, df_stdscaler, df_mmscaler

def create_2d_embeddings(df_noscaler, df_stdscaler, df_mmscaler):
    # first create embeddings for visualization, these have 2 dimensions only
    dfs={}
    dfs['all'] = df_noscaler
    dfs_mm={}
    dfs_mm['all'] = df_mmscaler
    dfs_std={}
    dfs_std['all'] = df_stdscaler

    # First 2 principal components
    dfs['pc2'] = pd.DataFrame(PCA(n_components=2).fit_transform(df_noscaler),
                    index=df_noscaler.index,
                    columns=["PC1", "PC2"])
    dfs_mm['pc2'] = pd.DataFrame(PCA(n_components=2).fit_transform(df_mmscaler),
                    index=df_mmscaler.index,
                    columns=["PC1", "PC2"])
    dfs_std['pc2'] = pd.DataFrame(PCA(n_components=2).fit_transform(df_stdscaler),
                    index=df_stdscaler.index,
                    columns=["PC1", "PC2"])
    
    return dfs, dfs_mm, dfs_std

def explained_variance_plot(df_noscaler, df_stdscaler, df_mmscaler):
    # explained variance plot for PCA
    pc=PCA()
    pc.fit(df_noscaler)
    expl_var=pc.explained_variance_ratio_.cumsum()
    pd.Series(expl_var, index=range(1, df_noscaler.shape[1]+1)).loc[:].plot(style=".-",
                                    label="expl. variance for PC with no scaling", legend=True)
    pc_mm=PCA()
    pc_mm.fit(df_mmscaler)
    expl_var_mm=pc_mm.explained_variance_ratio_.cumsum()
    pd.Series(expl_var_mm, index=range(1, df_mmscaler.shape[1]+1)).loc[:].plot(style=".-",
                                    label="expl. variance for PC with MinMax scaling", legend=True)
    pc_std=PCA()
    pc_std.fit(df_stdscaler)
    expl_var_std=pc_std.explained_variance_ratio_.cumsum()
    pd.Series(expl_var_std, index=range(1, df_stdscaler.shape[1]+1)).loc[:].plot(style=".-",
                                    label="expl. variance for PC with Standard scaling", legend=True)
    plt.savefig('figures/reduced_explained_variance_plot.png')
    return

def plot_PCA_projections(dfs, dfs_mm, dfs_std):
    f, ax = plt.subplots(1, 3, figsize=(15, 5))
    sns.scatterplot(x='PC1', y='PC2', data=dfs['pc2'], alpha=0.5, ax=ax[0]).set_title("PC projection")
    sns.scatterplot(x='PC1', y='PC2', data=dfs_mm['pc2'], alpha=0.5, ax=ax[1]).set_title("PC projection with MinMax")
    sns.scatterplot(x='PC1', y='PC2', data=dfs_std['pc2'], alpha=0.5, ax=ax[2]).set_title("PC projection with Standard")
    plt.savefig('figures/reduced_PCA_projection.png')
    return
if __name__ == '__main__':
    encoded_linkers = load_in_data() #currently loading in reduced dataset
    df_noscaler, df_stdscaler, df_mmscaler = make_pandas_dataframes(encoded_linkers)
    dfs, dfs_mm, dfs_std = create_2d_embeddings(df_noscaler, df_stdscaler, df_mmscaler)
#    explained_variance_plot(df_noscaler, df_stdscaler, df_mmscaler)
    plot_PCA_projections(dfs, dfs_mm, dfs_std)
