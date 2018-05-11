import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE

# # define the colormap
# cmap = plt.cm.jet
# # extract all colors from the .jet map
# cmaplist = [cmap(i) for i in range(cmap.10)]
# # create the new map
# cmap = cmap.from_list('Custom cmap', cmaplist, cmap.10)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def plot_mnist(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='gray')

    return fig

def plot_synthetic(samples, label = None, mode="simple"):
    fig = plt.figure(figsize=(4, 4))
    if mode == "simple":
        embeded = np.array(samples)
    elif mode == "tsne":
        embeded = TSNE().fit_transform(samples)
    else:
        print("unsupported plot mode")
        exit()
    plt.scatter(embeded[:,0], embeded[:,1], c = label, cmap='tab20c')
    return fig

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])
