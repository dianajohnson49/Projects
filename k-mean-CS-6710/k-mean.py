"""
HW 2: K-Mean Algorithm and Clustering
Diana Johnson
11-14-2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def combine_data(file1, file2):
    # class 0 data
    data1 = pd.read_csv(file1, delimiter=';')
    data1["label"] = 0

    # class 1 data
    data2 = pd.read_csv(file2, delimiter=';')
    data2["label"] = 1

    data = pd.concat([data1,data2])

    return data

def strat1(data):
    num_features = data.shape[1] - 1
    center = np.zeros(num_features)

    for i in range(num_features):
        feature_min = data.iloc[:,i].min()
        feature_max = data.iloc[:,i].max()

        center[i] = np.random.uniform(low=feature_min, high=feature_max)
    
    return center

def strat2(data):
    num_features = data.shape[1] - 1
    center = np.zeros(num_features)

    for i in range(num_features):
        sorted_vals = np.sort(data.iloc[:,1].values)

        # indeces for 1st and 5th bins
        end_bin_1 = len(sorted_vals) // 5
        start_bin_5 = 4 * end_bin_1

        # values between 20-80%
        new_vals = sorted_vals[end_bin_1: start_bin_5]
        feature_min = np.min(new_vals)
        feature_max = np.max(new_vals)

        center[i] = np.random.uniform(low=feature_min, high=feature_max)

    return center


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

    center = strat2(dataset)
    print(center)

# count # in each class
#numpy.bincount

main()

