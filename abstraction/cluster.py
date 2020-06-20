import sys
base_folder = 'E:/Dima/PhD/Repos/bert_verify/abstract_bert/'
sys.path.insert(1, base_folder)
from abstraction.ClusterEmbedding import ClusterEmbedding
import os

folder_hidden_states = f'{base_folder}/save/embeddings/'
fnames_hid_train = [folder_hidden_states + x for x in os.listdir(folder_hidden_states) if ('train' in x) & ('hidden' in x) ]

layer = 0
bs = 32
expected_size = (bs, 125, 768)

# merge embeddings per layer
for layer in range(0, 12):
    fout = f'{base_folder}/save/merged_embeddings/df_train_hid_{layer}.npy'
    ClusterEmbedding.merge_embeddings(fnames_hid_train, fout, layer, expected_size)
