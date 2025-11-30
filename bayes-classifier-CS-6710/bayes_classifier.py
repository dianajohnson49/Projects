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
    
    return predictions

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

    return predictions