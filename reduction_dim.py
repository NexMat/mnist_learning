import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

def main(X):

    plt.subplot(161)
    plt.imshow(X[1].reshape((28, 28)), cmap="gray")
    plt.title("Originale")

    X_transform_ = None

    # Variation du nombre de composantes
    for i in range(1, 6):
        
        nb_components = i * 50

        pca = PCA(n_components = nb_components)
        #pca.fit(X)

        # transformation de X
        X_transform = pca.fit_transform(X)

        # transformation inverse
        X_inverse_transform = pca.inverse_transform(X_transform)

        plt.subplot(160 + i + 1)
        plt.imshow(X_inverse_transform[1].reshape((28, 28)), cmap="gray")
        plt.title(str(nb_components))

        if i == 5:
            X_transform_ = X_transform.copy()

    plt.show()

    plt.subplot(121)
    plt.plot(pca.explained_variance_)
    plt.subplot(122)
    plt.plot(pca.explained_variance_ratio_)
    plt.show()

    
def repartition(X):

    pca = PCA(n_components = 250)

    # transformation de X
    X_transform = pca.fit_transform(X)

    plt.plot(X_transform[0], X_transform[1], 'b.')
    plt.show()


def setup(mnist):

    print(mnist.data.shape)
    print(mnist.target.shape)

    plt.imshow(mnist.data[1].reshape((28, 28)), cmap="gray")
    plt.show()

    print(mnist.target[1])

    
if __name__ == '__main__':

    mnist = fetch_mldata("MNIST original",
        data_home="/home/mathieu/Documents/ComputerVision/")

    nb_echantillon = 700
    
    # Les 700 premiers echantillons ne sont que des 0
    #print(np.unique(mnist.target[:nb_echantillon]))

    # que des 0
    X = mnist.data[:nb_echantillon]

    # pour avoir differents labels
    data_shuffled = mnist.data.copy()
    np.random.shuffle(data_shuffled)
    X_shuffled = data_shuffled[:nb_echantillon]

    #setup(mnist)
    main(X_shuffled)
    #repartition(X_shuffled)

