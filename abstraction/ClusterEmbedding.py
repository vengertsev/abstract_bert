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
    def cluster_embeddings():
        pass

    @staticmethod
    def get_clusters_stats():
        pass

    @staticmethod
    def visualize_clusters_stats():
        pass
