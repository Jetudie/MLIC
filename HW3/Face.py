import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.manifold import LocallyLinearEmbedding

# Load tif data
def load_data_tif(img_files):
    # Create a numpy array with size (128*128, 1)
    images = np.zeros((1, 128**2))
    for f in img_files:
        img = plt.imread(f)
        img = img.reshape((1, 128**2))
        images = np.vstack((images, img))
    images = np.delete(images, 0, 0)
    return images


# PCA using scikit learn
def my_pca(x, n_components):
    #x = np.swapaxes(x, 0, 1)
    pca = PCA(n_components = n_components)
    pca.fit(x)
    #return np.swapaxes(pca.transform(x), 0, 1)
    #return pca.transform(x)
    return pca


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


def mean_face(images):
    mean = np.mean(images, axis=0)
    plot_face(mean, 'mean_face')
    return mean


def plot_face(image, title):
    fig = plt.figure(figsize=(8,8))
    plt.title(title)
    plt.imshow(image.reshape(128, 128), cmap='gray')
    plt.savefig(title)
    plt.show()
    plt.close(fig)


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
    FILE_DIR = os.path.join(CURRENT_DIR, 'training.db')
    TRAIN_IMAGE_FILE = []
    for i in range(len(os.listdir(FILE_DIR))):
        TRAIN_IMAGE_FILE.append(os.path.join(FILE_DIR, os.listdir(FILE_DIR)[i]))

    images = load_data_tif(TRAIN_IMAGE_FILE)
    print("Shape of images:"+ str(np.shape(images)))

    mean_face = mean_face(images)
    print("Shape of mean face:"+ str(np.shape(mean_face)))

    images = images - mean_face
    eigenface_top_5 = my_pca(images, 9)
    for i in range(9):
        plot_face(eigenface_top_5.components_[i], 'top5_eigenface' + str(i))
    print("Shape of images:"+ str(np.shape(images)))
    print("Shape of top 5 eigenface:"+ str(np.shape(eigenface_top_5.components_)))
