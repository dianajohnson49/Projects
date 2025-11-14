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
        sorted_vals = np.sort(data.iloc[:,i].values)

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
    X = data.iloc[:, :-1].values

    inertia = 0.0
    for i in range(len(X)):
        centroid = centroids[labels[i]]
        inertia += np.sum((X[i] - centroid) ** 2)

    return inertia

def find_opt_clusters(data, k, centroids, max_iters = 200, tolerance = 1e-6):
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

    # remove label column
    X = np.array(data.iloc[:,:-1].values)
    
    centroids = np.array(centroids)

    for _ in range(max_iters):
        labels = np.zeros(len(X), dtype=int)

        # find closest centroid for each points
        for i in range(len(X)):
            dists = []
            for j in range(k):
                d = np.sum((X[i] - centroids[j]) ** 2)
                dists.append(d)
            
            # grabs index of smallest dist, which is the closest centroid
            labels[i] = np.argmin(dists)
        

        # initialize zeros matrix, matching current centroid shape
        new_centroids = np.zeros_like(centroids)

        # Re-compute centroids
        for j in range(k):
            # pull out each point that matches the label for the current cluster j
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                # get a mean for each feature
                new_centroids[j] = np.mean(cluster_points, axis=0)
            else:
                # keep old center
                new_centroids[j] = centroids[j]

        # euclidean dist of centroids
        if np.linalg.norm(new_centroids-centroids) < tolerance:
            break
        
        centroids = new_centroids

    return centroids, labels


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
    k_max = num_features-1

    inertia_1 = []
    
    for i in range(k_min,k_max+1):
        centroids = []
        for j in range(i):
            centroids.append(strat1(data))

        centroids, centroid_labels = find_opt_clusters(data, i, centroids)

        inertia = calculate_inertia(data,centroids,centroid_labels)
        inertia_1.append(inertia)

    
    inertia_2 = []
    
    for i in range(k_min,k_max+1):
        centroids = []
        for j in range(i):
            centroids.append(strat1(data))

        centroids, centroid_labels = find_opt_clusters(data, i, centroids)

        inertia = calculate_inertia(data,centroids,centroid_labels)
        inertia_2.append(inertia)
    
    # Plotting stuff
    plt.figure()

    plt.plot(range(k_min,k_max+1), inertia_1, marker='o', label='Strategy 1')
    plt.plot(range(k_min,k_max+1), inertia_2, marker='s', label='Strategy 2')
    plt.title("Elbow Method")
    plt.xlabel("k")
    plt.ylabel("Inertia")

    plt.xticks(range(k_min,k_max+1))

    plt.grid(True)
    plt.legend()
    plt.show()

def calculate_purity(labels, true_labels):
    """
    calculate_purity() : Computes the purity for a given set of clusters

    Parameters:
        labels: predicted cluster labels
        true_labels: true class labels

    Returns:
        Purity: float, calculated purity
    """

    labels = np.array(labels)
    true_labels = np.array(true_labels)
    # find number of classes
    cluster_ids = np.unique(labels)

    total_correct = 0
    for cluster in cluster_ids:
        # find points associated with current cluster
        cluster_points = true_labels[labels == cluster]

        # if no points assigned to cluster, move on
        if len(cluster_points) == 0:
            continue
        
        # how many points in the cluster have true labels
        counts = np.bincount(cluster_points)

        # find majority correct class
        majority = counts.max()

        # sum max overlap over all clusters
        total_correct += majority

    purity = total_correct / len(labels)

    return purity

def visualize_pca(data):
    """
    visualize_pca() : Performs PCA and chooses either 2 or 3 componenets to plot so
                    that cumulative variance >= 0.95. PLots a 2D or 3D plot depending on
                    number of components chosen.

    Parameters:
        data: pd dataframe, dataset of features for analysis

    Returns:
        None
    """
    X = data.iloc[:, :-1].values
    y = data["label"].values
    n = X.shape[0]

    X_mean_centered = X - np.mean(X, axis=0)

    # covariance matrix
    S = (X_mean_centered.T @ X_mean_centered) / (n-1)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(S)

    # sort eigenvectors for eigenvalue
    # argsort() returns indeces of sorted array
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # Normalize eigenvalues
    eigenvalues_norm = eigenvalues / np.sum(eigenvalues)

    # Determine if top 2 or 3 dimensions are needed for sum var >= 0.95
    cumulative_sum = np.cumsum(eigenvalues_norm)

    if cumulative_sum[1] >= 0.95:
        dim = 2
    else:
        dim = 3
    
    print(f"Selected PCA dimensions: {dim}")
    print(f"Cumulative normalized variance: {cumulative_sum[:dim]}")

    # grab necessary components
    components = eigenvectors[:, :dim]

    # Visualize the data
    X_pca = X_mean_centered @ components

    if dim == 2:
        plt.figure()
        for c in np.unique(y):
            plt.scatter(
                X_pca[y == c, 0],
                X_pca[y == c, 1],
                label=f"Class {c}",
                alpha=0.7,
            )

        # Plotting stuff
        plt.title("PCA Projection (2D) - Variance ≥ 95%")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)
        plt.legend()
        plt.show()
    else: 
    # dim = 3
        fig = plt.figure()
        # still one plot, but tells matplotlib 3d is needed
        ax = fig.add_subplot(111, projection="3d")

        # for each class, assign labels
        for c in np.unique(y):
            ax.scatter(
                X_pca[y == c, 0],
                X_pca[y == c, 1],
                X_pca[y == c, 2],
                label=f"Class {c}",
                alpha=0.7,
            )

        # PLotting stuff
        ax.set_title("PCA Projection (3D) - Variance ≥ 95%")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.legend()
        plt.show()
        


def main():
    """
    main() : Main driver method

    Parameters:
        None

    Returns:
        None
    """

    # Import datasets
    white_wine = "winequality-red.csv"
    red_wine = "winequality-white.csv"

    # Combine CSVs
    dataset = combine_data(white_wine, red_wine)

    # Use elbow method to find optimal number of clusters, k (FOUND: k=4)
    # Visualization of elbow method
    elbow(dataset)

    # PCA Analysis
    visualize_pca(dataset)
    
    # find the purity for each strategy
    k = 4

    # Strategy 1
    centroids1 = []
    for j in range(k):
        centroids1.append(strat1(dataset))

    centroids1, centroid1_labels = find_opt_clusters(dataset, k, centroids1)
    purity1 = calculate_purity(dataset['label'].values, centroid1_labels)
    print(f"Strategy 1 Purity (k=4): {purity1}")

    # Strategy 2
    centroids2 = []
    for j in range(k):
        centroids2.append(strat2(dataset))

    centroids2, centroid2_labels = find_opt_clusters(dataset, k, centroids1)
    purity2 = calculate_purity(dataset['label'].values, centroid2_labels)
    print(f"Strategy 2 Purity (k=4): {purity2}")


main()

