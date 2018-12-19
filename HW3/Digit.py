import os
import math
import random
import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.manifold import LocallyLinearEmbedding

# Load mnist data
def load_data_mnist(img_file, label_file):
    # Load labels
    with open(label_file, 'rb') as label_f:
        magic, n = struct.unpack('>II', label_f.read(8))
        labels = np.fromfile(label_f, dtype=np.uint8)
    # Load images
    with open(img_file, 'rb') as img_f:
        magic, num, rows, cols = struct.unpack('>IIII', img_f.read(16))
        images = np.fromfile(img_f, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

def sort_label(x, y):
    num_image = [ [] for i in range(10) ]
    for i in range(60000):
        num_image[y[i]].append(x[i])
    num_label = [ [] for i in range(10) ]
    for i in range(10):
        num_label[i] = [i]*len(num_image[i])
    num_image = np.array(num_image)
    num_label = np.array(num_label)
    return num_image, num_label


# PCA using scikit learn
def my_pca(x, n_components):
    pca = PCA(n_components = n_components)
    pca.fit(x)
    return pca.transform(x)


# ICA using scikit learn
def my_ica(x, n_components):
    ica = FastICA(n_components = n_components)
    return  ica.fit_transform(x)


# LLE using scikit learn
def my_lle(x, n_components):
    lle = LocallyLinearEmbedding(n_components = n_components)
    return  lle.fit_transform(x)


# Plot using matplotlib
def plot_scatter(x, labels, title, name):
    fig = plt.figure(figsize=(20,10))
    plt.title(title)
    plt.scatter(x[:,0], x[:,1], c=labels, edgecolor='none', alpha=0.5, cmap=plt.get_cmap('jet', 10), s=5)
    #plt.colorbar()
    plt.savefig(name+str(labels[0]))
    plt.close(fig)
    #plt.show()


# Show images with plot
def plot_mnist(X, y, X_embedded, name, min_dist=40000.0, shift=60):
    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes(frameon=False)
    plt.title("Two-dimensional embedding of handwritten digits with %s" % name)
    plt.setp(ax, xticks=(), yticks=())
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                    wspace=0.0, hspace=0.0)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, marker="x")

    if min_dist is not None:
        from matplotlib import offsetbox
        shown_images = np.array([[15., 15.]])
        indices = np.arange(X_embedded.shape[0])
        random.shuffle(indices)
        for i in indices[:5000]:
            dist = np.sum((X_embedded[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist:
                continue
            shown_images = np.r_[shown_images, [X_embedded[i]]]
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(X[i].reshape(28, 28), 
                                                cmap=plt.cm.gray_r), X_embedded[i]+shift)
            #circle = matplotlib.patches.Circle(X_embedded[i] , radius=10, color='r')
            #circle = plt.Circle(X_embedded[i], 50, color='r')
            plt.plot(X_embedded[i][0], X_embedded[i][1], 'ro', markerfacecolor = 'none')
            ax.add_artist(imagebox)
            #ax.add_artist(circle)
    #plt.colorbar()
    plt.savefig(name+str(y[0])+'_image')
    plt.close(fig)
    #plt.show()


def train(num_image, num_label):
    num_transformed = my_pca(num_image, 2)
    plot_scatter(num_transformed, num_label, num_label[0], 'PCA')
    plot_mnist(num_image, num_label, num_transformed, 'PCA', 40000.0, 60)

    num_transformed = my_ica(num_image, 2)
    plot_scatter(num_transformed, num_label, num_label[0], 'ICA')
    plot_mnist(num_image, num_label, num_transformed, 'ICA', 0.00002, 0.002)
    
    num_transformed = my_lle(num_image, 2)
    plot_scatter(num_transformed, num_label, num_label[0], 'LLE')
    plot_mnist(num_image, num_label, num_transformed, 'LLE', 0.00002, 0.002)


if __name__ == "__main__":
    CURRENT_DIR = os.getcwd()
    TRAIN_IMAGE_FILE = os.path.join(CURRENT_DIR, 'train-images-idx3-ubyte')
    TRAIN_LABEL_FILE = os.path.join(CURRENT_DIR, 'train-labels-idx1-ubyte')

    x_train, y_train = load_data_mnist(TRAIN_IMAGE_FILE, TRAIN_LABEL_FILE)

    num_image, num_label = sort_label(x_train, y_train)

    for i in range(10):
        train(num_image[i], num_label[i])