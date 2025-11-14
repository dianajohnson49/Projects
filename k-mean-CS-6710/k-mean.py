"""
HW 2: K-Mean Algorithm and Clustering
Diana Johnson
11-14-2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def combine_data(file1, file2):
    """
    combine_data(): Combines two CSV files so all data is in one pd dataframe
                    (each CSV is data for a different class)

    Parameters:
        file1: A CSV file containing data points for one class
        file2: A CSV file containing data points for one class

    Returns:
        data: pd dataframe, concatenated dataframe with all data points and class labels added
    """
    # class 0 data
    data1 = pd.read_csv(file1, delimiter=';')
    data1["label"] = 0

    # class 1 data
    data2 = pd.read_csv(file2, delimiter=';')
    data2["label"] = 1

    data = pd.concat([data1,data2])

    return data

def strat1(data):
    """
    strat1() : For each feature find out the minimum and maximum values. Then randomly
            select a value for each feature (using uniform distribution) from the min-max 
            range to initialize one complete centroid.

    Parameters:
        data: pd dataframe, dataset of features for analysis

    Returns:
        centroid: center for given data
    """
    num_features = data.shape[1] - 1
    centroid = np.zeros(num_features)

    for i in range(num_features):
        feature_min = data.iloc[:,i].min()
        feature_max = data.iloc[:,i].max()

        centroid[i] = np.random.uniform(low=feature_min, high=feature_max)
    
    return centroid

def strat2(data):
    """
    strat2() : Arranges each feature in ascending order and partitions the values in five 
            quartiles. Discards the first and last quartile to find the minimum and maximum 
            values from rest of the quartiles. Randomly selects a value for each feature 
            (using uniform distribution) from the min-max range to initialize one complete 
            centroid.

    Parameters:
        data: pd dataframe, dataset of features for analysis

    Returns:
        centroid: list, one randomized center
    """
    num_features = data.shape[1] - 1
    centroid = np.zeros(num_features)

    for i in range(num_features):
        sorted_vals = np.sort(data.iloc[:,1].values)

        # indeces for 1st and 5th bins
        end_bin_1 = len(sorted_vals) // 5
        start_bin_5 = 4 * end_bin_1

        # values between 20-80%
        new_vals = sorted_vals[end_bin_1: start_bin_5]
        feature_min = np.min(new_vals)
        feature_max = np.max(new_vals)

        centroid[i] = np.random.uniform(low=feature_min, high=feature_max)

    return centroid

def calculate_inertia(data, centroids, labels):
    """
    calculate_inertia() : Finds the total inertia values at different k's

    Parameters:
        data: pd dataframe, dataset of features for analysis
        centroids: list, optimal cluster centers
        labels: list, associated centroids for each data point

    Returns:
        inertia: float, total distortion
    """
    inertia = 0.0
    for i in range(len(data)):
        centroid = centroids[labels[i]]
        inertia += np.sum((data[i] - centroid) ** 2)

    return inertia

# TODO
def find_opt_clusters(data, k, centroids, max_iters = 200, tolerance = 1e-4):
    """
    find_opt_clusters() : Calculates the best centroids for a given set of data and given 
                        k value

    Parameters:
        data: pd dataframe, dataset of features for analysis
        k: int, number of clusters
        centroids: list, centers for clusters
        max_iters: maximum iterations for finding opt centers
        tolerance: target error/difference between iterations

    Returns:
        centroids: optimal centers per cluster
        centroid_labels: best center for each data point
    """


    return centroids, centroid_labels

#TODO
def elbow(data):
    """
    elbow() : Plots the inertias for different k's for the user to find the optimal
            number of clusters

    Parameters:
        data: pd dataframe, dataset of features for analysis

    Returns:
        None
    """
    num_features = data.shape[1] - 1
    k_min = 1
    k_max = 5 #num_features-1

    inertia_1 = []
    
    for i in range(k_min,k_max+1):
        centroids = []
        for j in range(i):
            centroids.append(strat1(data))

        centroids, centroid_labels = find_opt_clusters(data, k, centroids)
        #TODO find the ideal centroids here / labels


        inertia = calculate_inertia(data,centroids,centroid_labels)
        inertia_1.append(inertia)


def main():
    """
    main() : Main driver method

    Parameters:
        None

    Returns:
        None
    """
    white_wine = "winequality-red.csv"
    red_wine = "winequality-white.csv"
    dataset = combine_data(white_wine, red_wine)



# count # in each class
#numpy.bincount

# linalg.eigh -> does eigendecomposition
# do linalg.eigh(XXt) -> returns 1d array eigenvalues and 2d square array of eigenvectors
# take top 2 to visualize 
main()

