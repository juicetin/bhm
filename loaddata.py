#!/bin/python3

import numpy as np
import math
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.svm import SVC

# Import our ML algorithms
from ML.validation import cross_validate_algo
from ML.knn import kNN
from ML.random_forests import rf
from ML.logistic_regression import lr

# Algos
from sklearn import neighbors

# Load all the data
def load_data():
    bath_and_dom_lowres = np.load('data/bathAndDomLabel_lowres.npz')
    label = bath_and_dom_lowres['label']
    labelcounts = bath_and_dom_lowres['labelcounts']
    bath_locations = bath_and_dom_lowres['locations']
    features = bath_and_dom_lowres['features']
    
    querypoints_lowres = np.load('data/queryPoints_lowres_v2_.npz')
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

    features = np.array(features)
    labels = np.array(list(map(str, labels)))

    classifiers = [
            # neighbors.KNeighborsClassifier(n_neighbors=5),                  
            # LogisticRegression(),                                           
            # LogisticRegression(multi_class='multinomial', solver='lbfgs'), 
            # RandomForestClassifier(),                                       
            SVC()
            ]

    # 10-fold cross-validation for all
    for classifier in classifiers:
        cross_validate_algo(features, labels, 10, classifier)

    # Whiten
    features = scale(normalize(features))

    # # 10-fold cross-validation for all
    # for classifier in classifiers:
    #     cross_validate_algo(features, labels, 10, classifier)
