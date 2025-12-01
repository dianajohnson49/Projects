"""
HW 3: Gini Index
Diana Johnson
11-30-2025
"""

import pandas as pd
import numpy as np

def gini_impurity(labels):
    """
    gini_impurity(): Computes the Gini impurity of a set of class labels

    Parameters:
        labels: class labels for the samples in a subset of the data.

    Returns:
        impurity: float, Gini impurity value 
    """
    probs = labels.value_counts(normalize=True)
    impurity = 1 - np.sum(probs**2)

    return impurity

def gini_gain(data, attribute, target="Class"):
    """
    gini_gain(): Computes the Gini gain achieved by splitting a dataset on a given
                 attribute

    Parameters:
        data: pandas dataframe containing the dataset
        attribute: string, name of the attribute used to split the data
        target: string, name of the class label column (default = "Class")

    Returns:
        gain: float, the reduction in Gini impurity obtained by the split
    """
    total_gini = gini_impurity(data[target])
    num_samples = len(data)

    weighted_gini = 0
    for val, subset in data.groupby(attribute):
        weight = len(subset) / num_samples
        weighted_gini += weight * gini_impurity(subset[target])
    
    gain = total_gini - weighted_gini

    return gain

def main():
    """
    main() : Main driver method

    Parameters:
        None

    Returns:
        None
    """ 
    # Load the dataset
    data = pd.read_csv("customer_index.csv")

    attributes = ["Gender", "CarType", "CustomerID"]
    for a in attributes:
        gain = round(gini_gain(data, a),4)
        print(f"Gini Gain for {a}: {gain}")


if __name__ == "__main__":
    main()