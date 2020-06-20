import pandas as pd
import torch
import numpy as np
from numpy import save
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


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
        pca = PCA(n_components=120)
        pca.fit(x)
        print(f'pca.singular_values_={pca.singular_values_}')
        print(f'pca.singular_values_[0]={pca.singular_values_[0]}')
        print(f'pca.singular_values_[100]={pca.singular_values_[100]}')
        pca_keep = 100
        x2 = PCA(n_components=pca_keep).fit_transform(x)
        return x2

    @staticmethod
    def cluster_embeddings(fname_merged_embedding, fout, clustering_method):
        pass

    @staticmethod
    def get_clusters_stats():
        pass

    @staticmethod
    def visualize_clusters_stats():
        pass
