
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def read_data(path: str) -> pd.DataFrame:
    ''' Read in data from csv file '''
    df = pd.read_csv(path, index_col=0)
    df = df[['Line Name', 'Type', '24.0']] # Keep only columns EDD style
    df = df.rename(columns={'24.0': 'value'}) # Rename 24.0 to value
    return df

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    ''' Get data to the right format for analysis '''
    new_columns = df['Type'].unique()
    data = pd.DataFrame()
    data.index = df['Line Name'].unique()

    # add new columns to data
    for col in new_columns:
        data[col] = 0

    # fill in data
    for l in data.index:
        for c in new_columns:
            value = df[(df['Line Name'] == l) & (df['Type'] == c)]['value'].values
            data.loc[l, c] = value
    
    # drop OD column
    data.drop('Optical Density', axis=1, inplace=True)
    return data

def plot_corr_heatmap(df: pd.DataFrame) -> None:
    ''' Calculate corr matrix and plot heatmap '''
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr.columns)), corr.columns)
    for (i, j), z in np.ndenumerate(corr):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    plt.show()

def pca_analysis(df, n_components=2, plot_contr=False):
    ''' Perform PCA analysis and plot results'''
    pca = PCA(n_components=n_components)
    pca_df = pd.DataFrame(pca.fit_transform(df.drop('Limonene', axis=1)))
    pca_df.index = df.index
    plt.scatter(pca_df[0], pca_df[1], s=8, color='black')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.rcParams.update({'font.size': 8})
    for i, txt in enumerate(pca_df.index):
        plt.annotate(txt, (pca_df[0][i], pca_df[1][i]))
    plt.show()

    if plot_contr:
        plt.bar(range(1, 10), pca.explained_variance_ratio_)
        plt.xticks(range(1, 10))
        plt.xlabel('Principal Component')
        plt.ylabel('Proportion of Variance')
        plt.show()

def tsne_analysis(df, n_components=2, perplexity=12):
    ''' Perform TSNE analysis and plot results'''
    tsne = TSNE(n_components=n_components, perplexity=perplexity)
    tsne_df = pd.DataFrame(tsne.fit_transform(df.drop('Limonene', axis=1)))
    tsne_df.index = df.index
    # Plot TSNE1 vs TSNE2 with labels; control marker size
    plt.scatter(tsne_df[0], tsne_df[1], s=8, color='black')
    plt.xlabel('TSNE1')
    plt.ylabel('TSNE2')
    plt.rcParams.update({'font.size': 8})
    for i, txt in enumerate(tsne_df.index):
        plt.annotate(txt, (tsne_df[0][i], tsne_df[1][i]))
    plt.show()
