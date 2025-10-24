"""
HW 1: Linear Regression
Diana Johnson
10-24-2025
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def LinearRegression(X, y, alpha, epochs):
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
    # Check if normal equation can be applied
    if X.shape[1] == np.linalg.matrix_rank(X):
        print("Using Normal Equation Solver\n")
        NormalEquationSolver(X,y)

    # If not, use gradient descent
    else:
        print("Using Gradient Based Solver\n")
        GradientBasedSolver(X,y,alpha,epochs)


def NormalEquationSolver(X, y):
    """
    NormalEquationSolver() : Solves linear regression problem using the Normal Equation

    Parameters:
        X: data matrix
        y: outcome matrix (class labels)
    Returns:
        w: weight vector
    """
    # Calculating optimal weights
    w = np.linalg.inv((X.T @ X)) @ (X.T @ y)
    predictions = X @ w

    # Calculating RSS
    rss = 0.5 * np.sum((y-predictions)**2)

    # Print results
    print(f"Optimal Weight Vector: {w}")
    print(f"RSS: {rss}")

    return w
    


def GradientBasedSolver(X, y, alpha, epochs):
    """
    GradientBasedSolver() : Solves linear regression problem using Gradient Descent

    Parameters:
        X: data matrix
        y: class labels
        alpha: learning rate
        epochs: number of epochs
    Returns:
        w: weight vector
    """
    epochs = int(epochs)
    alpha = float(alpha)
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

    return w

def split_data(data):
    """
    split_data(): randomly splits a dataset 70:30 of train vs test data

    Parameters:
        data: Dataset to split

    Returns:
        train: Dataset of train data (70%)
        test: Dataset of test data (30%)
    """
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)
    num_samples = X.shape[0]

    train_ratio = 0.7
    train_size = int(train_ratio * num_samples)

    # use indices to shuffle data for random split
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # pull out indices for train and test
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    # find X and y datasets
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train, y_train, X_test, y_test


def add_bias(X):
    """
    add_bias(): Adds a column of ones at the beginning of the X matrix

    Parameters:
        X: Data

    Return:
        X_bias: X with added bias column
    """
    # create matrix of ones
    bias_col = np.ones((X.shape[0],1))

    # stack together
    X_bias = np.hstack([bias_col, X])

    return X_bias

def run_tests(X_test, y_test, weights):
    accuracy = 0
    rss = 0

    return accuracy, rss

def main():
    """
    main() : Main driver method
    """
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Correct format: 'python linear_regression.py data_file alpha epochs "
        "(FOR SPLITTING DATA): boolean (True or False)'")
        return

    data_file = sys.argv[1]
    data = pd.read_csv(data_file)
    alpha = sys.argv[2]
    epochs = sys.argv [3]

    split_flag = False
    if len(sys.argv) >= 5:
        if sys.argv[4] == "True":
            print("Data will be split into 70/30 of train/test")
            split_flag = True
        else:
            pass

    if split_flag:
        accuracy_list = []
        rss_list = []

        for i in range(10):
            print("Splitting data...")
            X_train, y_train, X_test, y_test = split_data(data)
            X_train = add_bias(X_train)
            final_weights = LinearRegression(X_train, y_train, alpha, epochs)

            accuracy, rss = run_tests(X_test,y_test, final_weights)
            accuracy_list.append(accuracy)
            rss_list.append(rss)

        mean_acc = sum(accuracy_list)/len(accuracy_list)
        st_dev = (pd.Series(accuracy_list)).std()
        print(f"Mean accuracy: {mean_acc} | Standard Deviation: {st_dev}")

            
    else:
        X = data.iloc[:,:-1].values
        X = add_bias(X)
        y = data.iloc[:, -1].values
        final_weights = LinearRegression(X, y, alpha, epochs)




main()