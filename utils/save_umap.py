import matplotlib.cm as cm
import matplotlib.pyplot as plt

from os.path import join


def save_umap(features, labels, seed):
    reducer = umap.UMAP(random_state=seed).fit(features)
    embedding = reducer.transform(labels)
    fig_umap = plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=test_true_labels[-1], cmap=cm.tab10)
    return fig_umap

