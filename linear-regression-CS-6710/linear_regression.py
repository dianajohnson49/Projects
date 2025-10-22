"""
HW 1: Linear Regression
Diana Johnson
10-24-2025
"""
import sys
import pandas as pd
import numpy as np

"""
Linear Regression

Parameters
    X: data matrix
    y: class labels
    alpha: learning rate
    epochs: number of epochs
Returns
    None
"""
def LinearRegression(X, y, alpha, epochs):
    # Check if normal equation can be applied
    if X.shape[1] == np.linalg.matrix_rank(X):
        print("Using Normal Equation Solver\n")
        NormalEquationSolver(X,y)

    # If not, use gradient descent
    else:
        print("Using Gradient Based Solver\n")
        GradientBasedSolver(X,y,alpha,epochs)

"""
NormalEquationSolver()

Parameters:
    X: data matrix
    y: outcome matrix (class labels)
Returns:
    None
"""
def NormalEquationSolver(X, y):
    # Calculating optimal weights
    w = np.linalg.inv((X.T @ X)) @ (X.T @ y)

    # Calculating RSS
    Residual_SS = (y - (X @ w)).T @ (y - (X @ w))

    # Print results
    print(f"Optimal Weight Vector: {w}")
    print(f"RSS: {Residual_SS}")
    

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


main()