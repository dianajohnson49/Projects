"""
HW 1: Linear Regression
Diana Johnson
10-24-2025
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
LinearRegression() : Checks if Normal Equation can be applied to a given input and calls
                    the Normal Equation Solver. If not, the Gradient Based Solver is called.

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
NormalEquationSolver() : Solves linear regression problem using the Normal Equation

Parameters:
    X: data matrix
    y: outcome matrix (class labels)
Returns:
    None
"""
def NormalEquationSolver(X, y):
    # Calculating optimal weights
    w = np.linalg.inv((X.T @ X)) @ (X.T @ y)
    predictions = X @ w

    # Calculating RSS
    rss = np.sum((y-predictions)**2)

    # Print results
    print(f"Optimal Weight Vector: {w}")
    print(f"RSS: {rss}")
    

"""
GradientBasedSolver() : Solves linear regression problem using Gradient Descent

Parameters:
    X: data matrix
    y: class labels
    alpha: learning rate
    epochs: number of epochs
Returns:
    None
"""
def GradientBasedSolver(X, y, alpha, epochs):
    num_samples, num_features = X.shape
    y = y.reshape(-1,1)
    w = np.zeros((num_features,1))

    # find cost of Wo
    predictions = X @ w
    current_cost = 0.5 * np.sum((y - predictions) ** 2)

    rss_list = [] 
    goal_cost = 10e-5

    print(f"Initial RSS: {current_cost}")
    
    for epoch in range(epochs):
        # Calculate the gradient
        gradient = (X.T @ X) @ w - X.T @ y

        # Update weights
        w = w - alpha * gradient

        # Calculate new RSS
        predictions = X @ w
        current_cost = 0.5 * np.sum((y - predictions) ** 2)

        rss_list.append(current_cost)

        print(f"Epoch {epoch + 1}: RSS = {current_cost}")

        # Check current cost/error
        if current_cost <= goal_cost:
            break
    
    print(f"Learned Weights: {w.flatten()}")

    plt.figure()

    plt.plot(range(1, len(rss_list)+1), rss_list)
    plt.title("Gradient Descent Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("RSS")

    plt.grid(True)
    plt.show()

def split_data(data_mat):
    pass

"""
main() : Main driver class
"""
def main():
    
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Correct format: 'python linear_regression.py data_file alpha epochs "
        "(FOR SPLITTING DATA): boolean (True or False)'")
        return

    data_file = sys.argv[1]
    data = pd.read_csv(data_file)
    alpha = sys.argv[2]
    epochs = sys.argv [3]
    
    if sys.argv[4]:
        if sys.argv[4] == "True":
            print("Splitting data into 70/30")
            split_data(data)
        else:
            pass

    X = data.iloc[:,:-1].values
    y = data.iloc[:, -1].values

    LinearRegression(X, y, alpha, epochs)


main()