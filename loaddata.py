#!/bin/python3

import numpy as np
import math
from sklearn import neighbors
from sklearn import cross_validation
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from pprint import pprint
import importlib

# Import our ML algorithms
from knn import kNN
from random_forests import rf
from logistic_regression import lr

# Load all the data
def load_data():
    bath_and_dom_lowres = np.load('./bathAndDomLabel_lowres.npz')
    label = bath_and_dom_lowres['label']
    labelcounts = bath_and_dom_lowres['labelcounts']
    bath_locations = bath_and_dom_lowres['locations']
    features = bath_and_dom_lowres['features']
    
    querypoints_lowres = np.load('./queryPoints_lowres_v2_.npz')
    qp_locations = querypoints_lowres['locations']
    validQueryID = querypoints_lowres['validQueryID']
    x_bins = querypoints_lowres['x_bins']
    query = querypoints_lowres['query']
    y_bins = querypoints_lowres['y_bins']

    return (label, labelcounts, bath_locations, features, querypoints_lowres, qp_locations, validQueryID, x_bins, query, y_bins)

# Main function
if __name__ == "__main__":
    # Load and assign all data
    labels, labelcounts, bath_locations, features, querypoints_lowres, qp_locations, validQueryID, x_bins, query, y_bins = load_data()

    # Comment everything out below and call functions accordingly using "python3 -i loaddata.py" to test algorithms interactively

    # kNN (all defaults) -  50 runs - 64.33%, 100 runs - 64.226%,  500 runs - 64.1495%, 1000 runs - 64.18585
    # kNN(features, labels, 10)

    # Random Forest (defaults) - 10 runs - 70.24%, 50 runs - 70.946%

    # Logistic Regression - (defaults, multinomial[lbfgs]) 10 runs - 94.166666666% every time
