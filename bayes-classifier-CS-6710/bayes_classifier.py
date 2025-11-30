"""
HW 3: Naive and Exact Bayes Classifier
Diana Johnson
11-30-2025
"""

import numpy as np
import pandas as pd

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

def load_weather_and_play_data(weather_path):
    """
    
    """
    df = pd.read_csv(weather_path)
    return df

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
        # get unique feature vectors/counts
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
    
    return predictions

def naive_bayes(X_train, y_train, X_test):
    """
    
    """
    X_train = X_train.astype(int)
    y_train = y_train.astype(int)

    classes = np.unique(y_train)= 

