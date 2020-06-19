from sklearn.cluster import KMeans
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

@st.cache
def data():
    #X = np.random.normal(0, 1, 1000).reshape(-1, 2)
    df_x = np.load('E:/Dima/PhD/Repos/bert_verify/abstract_bert/save/results/df_x.npy', mmap_mode='r')
    x = df_x[0:3200, :, :]  # 100 batches of size 32
    x = x.reshape(3200, -1)
    x2 = PCA(n_components=100).fit_transform(x)

    X = TSNE(n_components=2, perplexity=20, random_state=17).fit_transform(x2)

    return X


X = data()

cluster_slider = st.slider(
    min_value=1, max_value=6, value=2, label="Number of clusters: "
)
kmeans = KMeans(n_clusters=cluster_slider, random_state=0).fit(X)
labels = kmeans.labels_

selectbox = st.selectbox("Visualize confidence bounds", [False, True])
stdbox = st.selectbox("Number of standard deviations: ", [1, 2, 3])

clrs = ["red", "seagreen", "orange", "blue", "yellow", "purple"]

n_labels = len(set(labels))

individual = st.selectbox("Individual subplots?", [False, True])

if individual:
    fig, ax = plt.subplots(ncols=n_labels)
else:
    fig, ax = plt.subplots()

for i, yi in enumerate(set(labels)):
    if not individual:
        a = ax
    else:
        a = ax[i]

    xi = X[labels == yi]
    x_pts = xi[:, 0]
    y_pts = xi[:, 1]
    a.scatter(x_pts, y_pts, c=clrs[yi])

    if selectbox:
        confidence_ellipse(
            x=x_pts,
            y=y_pts,
            ax=a,
            edgecolor="black",
            facecolor=clrs[yi],
            alpha=0.2,
            n_std=stdbox,
        )
plt.tight_layout()
st.write(fig)
