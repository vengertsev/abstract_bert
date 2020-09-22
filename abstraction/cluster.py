import sys
base_folder = 'E:/Dima/PhD/Repos/bert_verify/abstract_bert/'
sys.path.insert(1, base_folder)
from abstraction.ClusterEmbedding import ClusterEmbedding
import os

# TODO: add logging instead of prints

folder_hidden_states = f'{base_folder}/save/embeddings/'

# BERT saved hidden space
fnames_hid_train = [folder_hidden_states + x for x in os.listdir(folder_hidden_states) if ('train' in x) & ('hid' in x)]

n_hid_layers = 12
bs = 32
expected_size = (bs, 125, 768)
sample_size = bs*100  # number of Twitts that are analysed (max is ~13200)
dim_red_method = 'pca'
clustering_method = 'dbscan'

# # merge embeddings per layer
# for layer in range(8, n_hid_layers):
#     fout = f'{base_folder}/save/merged_embeddings/df_train_hid_{layer}.npy'
#     ClusterEmbedding.merge_embeddings(fnames_hid_train, fout, layer, expected_size)

# reduce dimension
pca_keep = 1000
for layer in range(0, n_hid_layers):
    fname_merged_embedding = f'{base_folder}/save/merged_embeddings/df_train_hid_{layer}.npy'
    fout = f'{base_folder}/save/clusters/df_{dim_red_method}_{layer}_{pca_keep}.npy'
    ClusterEmbedding.reduce_dimension(fname_merged_embedding, fout, sample_size, dim_red_method, pca_keep)

# clustering
for layer in range(0, n_hid_layers):
    print(f'============= layer={layer} ===============')
    fname_dim_reduced = f'{base_folder}/save/clusters/df_{dim_red_method}_{layer}_{pca_keep}.npy'
    fout = f'{base_folder}/save/clusters/df_{clustering_method}_{layer}.npy'
    cluster_labels = ClusterEmbedding.cluster_embeddings(fname_dim_reduced, fout, clustering_method)
    fout = f'{base_folder}/save/clusters/{clustering_method}_{layer}.png'
    ClusterEmbedding.visualize_clusters(cluster_labels, fout, fname_dim_reduced)

# visualization is done in the app

