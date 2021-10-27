import itertools
import matplotlib.pyplot as plt
import numpy as np

from os.path import join
from sklearn.metrics import confusion_matrix

def save_confusion_matrix(labels, preds):
    img_name = 'confusion_matrix.jpg'
    # making confusion matrix
    # label_name=['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    label_name = list(range(10))
    conf_matrix = confusion_matrix(labels, preds, labels=label_name, normalize='true')
    # plot and save
    fig_conf_matrix = plt.figure()
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(label_name))
    plt.xticks(tick_marks, label_name, rotation=45)
    plt.yticks(tick_marks, label_name)

    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], '.2f'),
                 horizontalalignment='center',
                 color='white' if conf_matrix[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.xlabel('Pred Label')
    plt.ylabel('Ground Truth')

    return fig_conf_matrix
