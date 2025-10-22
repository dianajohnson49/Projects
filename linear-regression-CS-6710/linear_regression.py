"""
HW 1: Linear Regression
Diana Johnson
10-24-2025
"""
import sys
import pandas as pd
import numpy as np

"""

"""
def LinearRegression(X, y):
    pass

def GradientBasedSolver(X, y, alpha, epochs):
    pass


def main():
    # TODO: Replace with command line arguments
    """
    data_file = sys.argv[1]
    data = pd.read_csv(data_file)
    alpha = sys.argv[2]
    epochs = sys.argv [3]
    """
    data = pd.read_csv("sample_data.csv")
    alpha = 0.02
    epochs = 100

    X = data.iloc[:,:-1].values
    y = data.iloc[:, -1].values

    if X.shape[1] == np.linalg.matrix_rank(X):
        LinearRegression(X,y)
    else:
        GradientBasedSolver(X,y,alpha,epochs)

main()