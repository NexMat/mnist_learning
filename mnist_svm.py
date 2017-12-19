import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import matplotlib.pyplot as plt

def main(mnist):

    nb_components = 50

    # Splitting into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.33)

    # Scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    # Training on raw data
    print("\nTraining on raw data...")
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    print("Scoring on raw data...")
    score = clf.score(X_test, y_test)

    print("[!] Prediction on raw data:", score)

    for nb_components in [1, 50, 100, 150, 200, 250, 300]:

        print("\nNumber of components:", nb_components)
        # Dimension reduction
        pca = PCA(n_components = nb_components)
        pca.fit(X_train)

        # Transformation of data set (X)
        X_train_transform = pca.transform(X_train)
        X_test_transform = pca.transform(X_test)

        # Training on the pca data
        print("Training on PCA data...")
        clf2 = LinearSVC()
        clf2.fit(X_train_transform, y_train)
        print("Scoring on PCA data...")
        score_pca = clf2.score(X_test_transform, y_test)

        print("[!] Prediction on PCA data:", score_pca)
        print("-------------------------------")


if __name__ == '__main__':

    mnist = fetch_mldata("MNIST original",
        data_home="/home/mathieu/Documents/ComputerVision/")

    main(mnist)

