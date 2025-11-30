"""
HW 3: Naive and Exact Bayes Classifier
Diana Johnson
11-30-2025
"""

import numpy as np
import pandas as pd
from linear_regression import LinearRegression, split_data, add_bias, run_tests

# Import datasets
def load_zoo_data(zoo_path):
    """
    
    """
    col_names = [
        "animal_name", "hair", "feathers", "eggs", "milk", "airborne",
        "aquatic", "predator", "toothed", "backbone", "breathes",
        "venomous", "fins", "legs", "tail", "domestic", "catsize", "type"
    ]

    df = pd.read_csv(zoo_path, header=None, names=col_names)
    df = df.drop(columns=["animal_name"])
    return df

def load_weather_data(weather_path):
    """
    
    """
    df = pd.read_csv(weather_path)
    return df


def split_zoo_data(data, ratio=0.2):
    """
    split_zoo_data(): randomly splits a dataset 80:20 of train vs test data, making sure 
                    at least one sample per class is in the test set

    Parameters:
        data: Dataset to split

    Returns:
        X_train: Dataset of X train data (80%)
        y_train: Dataset of y test data (20%)
        X_test: Dataset of X test data (20%)
        y_test: Dataset of y test data (20%)
    """
    classes = data['type'].unique()
    test_idx = []

    for c in classes:
        class_idx = data[data['type'] == c].index
        np.random.shuffle(class_idx)
        # grab one sample per class for test set
        test_idx.append(class_idx[0])
    
    # grab rest of samples for test set
    # remove already chosen test samples
    remaining_idx = data.index.different(test_idx)
    # find num samples needed to meet ratio
    num_samples_needed = int(ratio * len(data)) - len(test_idx)
    # shuffle the remaining samples and select num needed
    np.random.shuffle(remaining_idx)
    test_idx.extend(remaining_idx[:num_samples_needed])

    train_idx = data.index.difference(test_idx)

    # pull out relevant feature information and convert to numpy array using .values
    X_train = data.loc[train_idx].drop(columns=['type']).values
    y_train = data.loc[train_idx, 'type'].values
    X_test = data.loc[test_idx].drop(columns=['type']).values
    y_test = data.loc[test_idx, 'type'].values

    return X_train, y_train, X_test, y_test

def split_weather_data(data, ratio=0.2):
    """
    split_weather_data(): randomly splits a dataset 80:20 of train vs test data

    Parameters:
        data: Dataset to split

    Returns:
        X_train: Dataset of X train data (80%)
        y_train: Dataset of y test data (20%)
        X_test: Dataset of X test data (20%)
        y_test: Dataset of y test data (20%)
    """
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    num_samples = X.shape[0]
    # use indices to shuffle data for random split
    indices = np.arrange(num_samples)
    np.random.shuffle(indices)

    # pull out indices for train and test
    test_size = int(ratio * num_samples)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    # find X and y datasets
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train, y_train, X_test, y_test


def exact_bayes(X_train, y_train, X_test):
    """
    
    """
    X_train = X_train.astype(int)
    y_train = y_train.astype(int)

    classes = np.unique(y_train)
    model = {}

    # full joint probability for given class
    for c in classes:
        # find all samples in train set that are class c
        Xc = X_train[y_train == c]
        # get unique feature vectors/counts for unique rows
        patterns, counts = np.unique(Xc, axis = 0, return_counts=True)
        model[c] = (patterns, counts, len(Xc))
    
    predictions = []

    # find prediction label for each test sample
    for x in X_test:
        max_prob = -1
        max_class = None

        for c, (patterns, counts, num_samples) in model.items():
            prob = 0
            # zip combines element by element for multiple iterable inputs
            for p, cnt in zip(patterns, counts):
                # check for matches
                if np.array_equal(p,x):
                    prob = cnt / num_samples
                    break
                
                # assign prediction to test sample
                if prob > max_prob:
                    max_prob = prob
                    max_class = c

        predictions.append(max_class)
    
    return np.array(predictions)

def naive_bayes(X_train, y_train, X_test):
    """
    
    """
    X_train = X_train.astype(int)
    y_train = y_train.astype(int)

    classes = np.unique(y_train)
    class_counts = {}

    # get num of samples for each class
    for c in classes:
        count = np.sum(y_train == c)
        class_counts[c] = count
    
    cond_prob = {}
    for cls in classes:
        Xc = X_train[y_train == cls]
        cond_prob[cls] = {}

        # loop through all features
        for feature in range(X_train.shape[1]):
            vals, counts = np.unique(Xc[:,feature], return_counts=True)

            probs = {}
            # get indeces and values for class c
            for i, v in enumerate(vals):
                count = counts[i]
                probability = count / len(Xc)
                probs[v] = probability
            
            cond_prob[cls][feature] = probs
        
    num_samples = len(y_train)
    predictions = []

    # for each test sample, assign prediction
    for x in X_test:
        max_prob = -1
        max_class = None

        for clas in classes:
            prob = class_counts[clas] / num_samples
            
            for i, val in enumerate(x):
                # for specific class c, and feature i, get the prob
                # if val not found, use small value so it doesn't mult by 0
                prob *= cond_prob[clas][i].get(val, 1e-9)
            
            if prob > max_prob:
                max_prob = prob
                max_class = clas

        predictions.append(max_class)

    return np.array(predictions)

def calculate_accuracy(y_true, y_pred):
    """
    
    """
    accuracy = np.mean(y_true == y_pred)
    return accuracy

def main():
    zoo_data = load_zoo_data("zoo/zoo.data")
    weather_data = load_weather_data("WeatherAndPlayData.txt")




main()