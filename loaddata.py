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

from matplotlib import pyplot as pl

# Import our ML algorithms
from ML.validation import cross_validate_algo
from ML.knn import kNN
from ML.random_forests import rf
from ML.logistic_regression import lr

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

def show_map(labels, locations):
    x_min = min(locations[:,0])
    x_max = max(locations[:,0])
    y_min = min(locations[:,1])
    y_max = max(locations[:,1])

    my_cmap = copy.copy(pl.cm.get_cmap('gray')) # get a copy of gray color map
    my_cmap.set_bad(alpha=1)                     # set how colourmap handles 'bad' values

    blank = [[50]] * len(locations)
    pl.imshow(blank, extent=[x_min, x_max, y_min, y_max], interpolation='nearest', cmap=my_cmap)

    # Plot points which were surveyed
    pl.plot(locations[:,0], locations[:,1], 'ro')
    pl.show()

# Main function
if __name__ == "__main__":
    # Load and assign all data
    # labels, labelcounts, bath_locations, features, querypoints_lowres, qp_locations, validQueryID, x_bins, query, y_bins = load_data()
    # labels, labelcounts, bath_locations, features = load_training_data()
    # qp_locations, validQueryID, x_bins, query, y_bins = load_test_data()

    # features = np.array(features)
    # labels = np.array(list(map(str, labels)))

    # classifiers = [
    #         # neighbors.KNeighborsClassifier(n_neighbors=5),                  
    #         # LogisticRegression(),                                           
    #         # LogisticRegression(multi_class='multinomial', solver='lbfgs'), 
    #         RandomForestClassifier(),                                       
    #         # SVC()
    #         ]

    from ML.gp import GaussianProcess
    # from sklearn.gaussian_process import GaussianProcess
    gp = GaussianProcess()
    X = np.array([-1.50,-1.00,-0.75,-0.40,-0.25,0.00])
    X = X.reshape(len(X), 1)
    y = np.array([-1.70,-1.20,-0.25,0.30,0.5,0.7])
    y = y.reshape(len(y), 1)
    x = np.array([0.2, 0.3, 0.4, 0.5])
    x = x.reshape(len(x), 1)
    gp.fit(X, y)

    # kss = gp.K_create(x, x)
    # ks = gp.K_create(x, X)
    # mean = gp.mean_func(X, x, y)
    y_pred, MSE = gp.predict(x)
    sigma = np.sqrt(MSE)
    # var = gp.variance(

    # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2)
    # rf = RandomForestClassifier()
    # y_ = rf.fit(X_train, y_train).predict(X_test)

    # show_map(labels, bath_locations)

    # 10-fold cross-validation for all
    # results = []
    # for classifier in classifiers:
    #     results.append(cross_validate_algo(features, labels, 10, classifier))

    # Whiten
    # features = scale(normalize(features))

    # # 10-fold cross-validation for all
    # for classifier in classifiers:
    #     cross_validate_algo(features, labels, 10, classifier)

    # Plot function, prediction, and 95% confidence interval based on MSE
    # confidence = 1.9600
    # fig = pl.figure()
    # # pl.plot(x, y, 'r:', label=u'$f(x) = x\, \sin(x)$')
    # pl.plot(X, y, 'r.', markersize=10, label=u'Observations')
    # pl.plot(x, y_pred, 'b-', label=u'Prediction')
    # pl.fill(np.concatenate([x, x[::-1]]),
    #         np.concatenate([y_pred - confidence * sigma,
    #                       (y_pred + confidence * sigma)[::-1]]),
    #         alpha=.5, fc='b', ec='None', label='95% confidence interval')
    # pl.xlabel('$x$')
    # pl.ylabel('$y$')
    # pl.show()
