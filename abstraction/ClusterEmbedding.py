import pandas as pd
import torch
import numpy as np
from numpy import save
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


class ClusterEmbedding:
    @staticmethod
    def merge_embeddings(fnames, fout, layer, expected_size):
        df_x = np.empty((0, 125, 768), float)

        for i in range(0, len(fnames)):
            if i % 10 == 0:
                print(i)
            fname = fnames[i]
            df = torch.load(fname)
            df_i = df[layer].detach().numpy()
            if df_i.shape == expected_size:
                df_x = np.append(df_x, df_i, axis=0)
        save(fout, df_x)

    @staticmethod
    def reduce_dimension(fname_merged_embedding, fout, sample_size, dim_red_method):
        df_x = np.load(fname_merged_embedding, mmap_mode='r')
        x = df_x[0:sample_size, :, :]
        print(f'sampled input to dim reduction:{x.shape}')

        if dim_red_method == 'pca':
            x = x.reshape(sample_size, -1)
            df = ClusterEmbedding._reduce_dim_pca(x)
            save(fout, df)

    @staticmethod
    def _reduce_dim_pca(x):
        x = StandardScaler().fit_transform(x)
        pca = PCA(n_components=120)
        pca.fit(x)
        print(f'pca.singular_values_={pca.singular_values_}')
        print(f'pca.singular_values_[0]={pca.singular_values_[0]}')
        print(f'pca.singular_values_[100]={pca.singular_values_[100]}')
        pca_keep = 100
        x2 = PCA(n_components=pca_keep).fit_transform(x)
        return x2

    @staticmethod
    def cluster_embeddings(fname, fout, clustering_method):
        df_x = np.load(fname)
        if clustering_method == 'dbscan':
            cluster_labels = ClusterEmbedding._cluster_dbscan(df_x)
            return cluster_labels


    @staticmethod
    def _cluster_dbscan(x):
        # TODO: tune epsilon
        # TODO: distance histogram and find quantiles for epsilon - try all metrics sklearn.metrics.pairwise
        # TODO: find the right metric
        clustering = DBSCAN(eps=25, min_samples=5).fit(x)
        unique_elements, counts_elements = np.unique(clustering.labels_, return_counts=True)
        print("Frequency of unique values of the said array:")
        print(np.asarray((unique_elements, counts_elements)))
        return clustering.labels_

    @staticmethod
    def get_clusters_stats():
        pass

    @staticmethod
    def _pltcolor(lst):
        cols = []
        for l in lst:
            if l == 0:
                cols.append('red')
            elif l == 1:
                cols.append('blue')
            elif l == 2:
                cols.append('green')
            elif l == 3:
                cols.append('yellow')
            elif l == 4:
                cols.append('c')
            elif l == 5:
                cols.append('lime')
            elif l == 6:
                cols.append('slateblue')
            elif l == 7:
                cols.append('darkorange')
            elif l == 8:
                cols.append('lightcoral')
            elif l == 9:
                cols.append('teal')
            elif l == 10:
                cols.append('olive')
            else:
                cols.append('black')
        return cols

    @staticmethod
    def visualize_clusters(cluster_labels, fout, fname):
        x = np.load(fname)
        colors = ClusterEmbedding._pltcolor(cluster_labels)
        x3 = TSNE(n_components=2, perplexity=100, random_state=17).fit_transform(x)
        plt.scatter(x3[:, 0], x3[:, 1], c=colors)
        plt.savefig(fout, dpi=150)
