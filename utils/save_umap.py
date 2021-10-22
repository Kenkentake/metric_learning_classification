import matplotlib.cm as cm
import matplotlib.pyplot as plt
import umap

from os.path import join


def save_umap(features, labels, seed):
    reducer = umap.UMAP(random_state=seed).fit(features)
    embedding = reducer.transform(features)
    fig_umap = plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=cm.tab10)
    plt.colorbar()
    return fig_umap
