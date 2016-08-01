#!/bin/python3

import numpy as np
import copy
import math
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import LeaveOneOut
from sklearn.cross_validation import StratifiedKFold
from sklearn import datasets
from datetime import datetime

from matplotlib import pyplot as plt

# Import our ML algorithms
from ML.validation import cross_validate_algo
from ML.knn import kNN
from ML.random_forests import rf
from ML.logistic_regression import lr
from ML.gp import GaussianProcess

import visualisation as vis

# Algos
from sklearn import neighbors

# Load all the data
def load_training_data():
    bath_and_dom_lowres = np.load('data/bathAndDomLabel_lowres.npz')
    label = bath_and_dom_lowres['label']
    labelcounts = bath_and_dom_lowres['labelcounts']
    bath_locations = bath_and_dom_lowres['locations']
    features = bath_and_dom_lowres['features']
    
    return (label, labelcounts, bath_locations, features)

def load_test_data():
    querypoints_lowres = np.load('data/queryPoints_lowres_v2_.npz')
    qp_locations = querypoints_lowres['locations']
    validQueryID = querypoints_lowres['validQueryID']
    x_bins = querypoints_lowres['x_bins']
    query = querypoints_lowres['query']
    y_bins = querypoints_lowres['y_bins']

    return (qp_locations, validQueryID, x_bins, query, y_bins)

def stratified_micro_batch(features, labels, point_count):
    sss = StratifiedShuffleSplit(labels, 1, test_size=point_count/len(features))
    for train_index, test_index in sss:
        pass
    return features[test_index], labels[test_index]

def summarised_labels(labels):
    label_map={1:0,2:0,3:1,4:3,5:1,6:3,7:3,8:3,9:3,10:1,11:3,12:3,13:2,14:2,15:2,16:1,17:1,18:0,19:1,20:0,21:0,22:1,23:0,24:0}
    new_labels = np.copy(labels)
    for k, v in label_map.items(): 
        new_labels[labels==k] = v
    return new_labels

def classification_bathy_testing(features, labels):
    ####################################################################################
    # GP Classification on Bathymetry Data
    ####################################################################################

    gp = GaussianProcess()

    # cv = LeaveOneOut(len(labels))
    cv = StratifiedKFold(labels, n_folds=10)
    # cv = StratifiedShuffleSplit(labels, 1, test_size=4/len(features))
    accuracies = []
    for train_index, test_index in cv:
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        gp.fit_class(X_train, y_train)
        y_pred = gp.predict_class(X_test)

        accuracy = gp.score(y_pred, y_test)
        accuracies.append(accuracy)
        print("Accuracy is: {}".format(accuracy))

        print("P: {}\n, A: {}".format(y_pred, y_test))

    print("Average accuracy: {}".format(np.average(accuracies)))

def regression_dummy_testing():
    ####################################################################################
    # Dummy testing
    ####################################################################################

    a = np.array([i for i in range(1,10)]).reshape(3,3)
    b = np.array([2,2,2])

    from ML.gp import GaussianProcess
    # from sklearn.gaussian_process import GaussianProcess
    gp = GaussianProcess()
    X = np.array([-1.50,-1.00,-0.75,-0.40,-0.25,0.00])
    X = X.reshape(len(X), 1)
    y = np.array([-1.70,-1.20,-0.25,0.30,0.5,0.7])
    # y = y.reshape(len(y), 1)

    # x = np.array([-1.45, -1.32, -0.5, -0.32, -0.10, 0.2, 0.3, 0.4])
    x = np.array([0.2])
    x = x.reshape(len(x), 1)
    gp.fit(X, y)

    y_pred, MSE = gp.predict(x)
    sigma = np.sqrt(MSE)

    # Plot function, prediction, and 95% confidence interval based on MSE
    from visualisation import plot
    # plot(X, y, x, y_pred, sigma)

def classification_dummy_testing():
    ####################################################################################
    # Dummy testing - classification
    ####################################################################################

    gp = GaussianProcess()

    #X, y = datasets.make_circles(n_samples=12)
    # n_informative
    # n_redundant
    # n_repeated
    num = 3
    X, y = datasets.make_classification(n_samples=1000,
            n_features=num, 
            n_clusters_per_class=2,
            n_redundant=0, 
            n_repeated=0,
            n_informative=num,
            n_classes=num)

    from sklearn.cross_validation import StratifiedKFold
    cv = StratifiedKFold(y, n_folds = 20)
    # cv = StratifiedShuffleSplit(y, 1, test_size=33/len(X))
    accuracies = []
    for train_index, test_index in cv:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        gp.fit_class(X_train, y_train)
        y_pred = gp.predict_class(X_test)

        accuracy = gp.score(y_pred, y_test)
        accuracies.append(accuracy)
        print("Accuracy is: {}".format(accuracy))

    print("Average accuracy: {}".format(np.average(accuracies)))

# Main function
if __name__ == "__main__":
    # Load and assign all data
    # labels, labelcounts, bath_locations, features, querypoints_lowres, qp_locations, validQueryID, x_bins, query, y_bins = load_data()

    print("Loading data from npzs...")
    labels, labelcounts, bath_locations, features = load_training_data()
    qp_locations, validQueryID, x_bins, query, y_bins = load_test_data()

    print("Extracting features/etc...")
    features = np.array(features)
    features = scale(normalize(features)) # Whiten
    labels = np.array(labels)
    # labels = summarised_labels(labels)

    mini_features, mini_labels = stratified_micro_batch(features, labels, 300)

    # x_bins_training, y_bins_training = np.meshgrid(list(set(bath_locations[:,0])), list(set(bath_locations[:,1])))
    # x_bins_training, y_bins_training = list(set(bath_locations[:,0])), list(set(bath_locations[:,1]))

    # classifiers = [
    #         # neighbors.KNeighborsClassifier(n_neighbors=5),                  
    #         # LogisticRegression(),                                           
    #         # LogisticRegression(multi_class='multinomial', solver='lbfgs'), 
    #         RandomForestClassifier(),                                       
    #         # SVC()
    #         ]

    # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2)
    # rf = RandomForestClassifier()
    # y_ = rf.fit(X_train, y_train).predict(X_test)

    # Visualisation
    # vis.show_map(qp_locations, query[:,3], x_bins, y_bins)
    # vis.show_map(bath_locations, labels, x_bins_training, y_bins_training)

    # 10-fold cross-validation for all
    # results = []
    # for classifier in classifiers:
    #     results.append(cross_validate_algo(features, labels, 10, classifier))

    # # 10-fold cross-validation for all
    # for classifier in classifiers:
    #     cross_validate_algo(features, labels, 10, classifier)

    classification_bathy_testing(mini_features, mini_labels)
    # classification_dummy_testing():
