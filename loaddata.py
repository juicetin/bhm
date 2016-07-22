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

from matplotlib import pyplot as plt

# Import our ML algorithms
from ML.validation import cross_validate_algo
from ML.knn import kNN
from ML.random_forests import rf
from ML.logistic_regression import lr

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


# Main function
if __name__ == "__main__":
    # Load and assign all data
    # labels, labelcounts, bath_locations, features, querypoints_lowres, qp_locations, validQueryID, x_bins, query, y_bins = load_data()
    labels, labelcounts, bath_locations, features = load_training_data()
    qp_locations, validQueryID, x_bins, query, y_bins = load_test_data()

    features = np.array(features)
    labels = np.array(labels)
    # x_bins_training, y_bins_training = np.meshgrid(list(set(bath_locations[:,0])), list(set(bath_locations[:,1])))
    x_bins_training, y_bins_training = list(set(bath_locations[:,0])), list(set(bath_locations[:,1]))

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

    # vis.show_map(labels, qp_locations, x_bins, y_bins)
    Z = vis.show_map(bath_locations, labels, x_bins_training, y_bins_training)

    # 10-fold cross-validation for all
    # results = []
    # for classifier in classifiers:
    #     results.append(cross_validate_algo(features, labels, 10, classifier))

    # Whiten
    # features = scale(normalize(features))

    # # 10-fold cross-validation for all
    # for classifier in classifiers:
    #     cross_validate_algo(features, labels, 10, classifier)

    ####################################################################################
    # Dummy testing
    ####################################################################################

    # a = np.array([i for i in range(1,10)]).reshape(3,3)
    # b = np.array([2,2,2])

    # from ML.gp import GaussianProcess
    # # from sklearn.gaussian_process import GaussianProcess
    # gp = GaussianProcess()
    # X = np.array([-1.50,-1.00,-0.75,-0.40,-0.25,0.00])
    # X = X.reshape(len(X), 1)
    # y = np.array([-1.70,-1.20,-0.25,0.30,0.5,0.7])
    # # y = y.reshape(len(y), 1)

    # # x = np.array([-1.45, -1.32, -0.5, -0.32, -0.10, 0.2, 0.3, 0.4])
    # x = np.array([0.2])
    # x = x.reshape(len(x), 1)
    # gp.fit(X, y)

    # y_pred, MSE = gp.predict(x)
    # sigma = np.sqrt(MSE)

    # # Plot function, prediction, and 95% confidence interval based on MSE
    # from visualisation import plot
    # # plot(X, y, x, y_pred, sigma)

    ####################################################################################
    # Dummy testing - classification
    ####################################################################################

    # from ML.gp import GaussianProcess
    # gp = GaussianProcess()

    # from sklearn import datasets
    # from datetime import datetime

    # #X, y = datasets.make_circles(n_samples=12)
    # X, y = datasets.make_classification(n_samples=100,
    #         n_features=2, 
    #         n_clusters_per_class=1,
    #         n_redundant=0, 
    #         n_repeated=0,
    #         n_informative=1,
    #         n_classes=2)

    # cmaps = [('Perceptually Uniform Sequential',
    #                             ['viridis', 'inferno', 'plasma', 'magma']),
    #          ('Sequential',     ['Blues', 'BuGn', 'BuPu',
    #                              'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
    #                              'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
    #                              'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
    #          ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',
    #                              'copper', 'gist_heat', 'gray', 'hot',
    #                              'pink', 'spring', 'summer', 'winter']),
    #          ('Diverging',      ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
    #                              'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
    #                              'seismic']),
    #          ('Qualitative',    ['Accent', 'Dark2', 'Paired', 'Pastel1',
    #                              'Pastel2', 'Set1', 'Set2', 'Set3']),
    #          ('Miscellaneous',  ['gist_earth', 'terrain', 'ocean', 'gist_stern',
    #                              'brg', 'CMRmap', 'cubehelix',
    #                              'gnuplot', 'gnuplot2', 'gist_ncar',
    #                              'nipy_spectral', 'jet', 'rainbow',
    #                              'gist_rainbow', 'hsv', 'flag', 'prism'])]

    # import pylab
    # from matplotlib import pyplot as plt
    # # colours = [int(i % 23) for i in range(100)]
    # # print(colours)
    # plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.get_cmap('ocean'))
    # plt.show()

    # # d1 = datetime.now()

    # # from sklearn.cross_validation import StratifiedKFold
    # # kf = StratifiedKFold(y, n_folds = 10)
    # # accuracies = []
    # # for train_index, test_index in kf:
    # #     X_train, X_test = X[train_index], X[test_index]
    # #     y_train, y_test = y[train_index], y[test_index]

    # #     gp.fit_class(X_train, y_train)
    # #     y_pred = gp.predict_class(X_test)

    # #     accuracy = gp.score(y_pred, y_test)
    # #     accuracies.append(accuracy)
    # #     print("Accuracy is: {}".format(accuracy))

    # # print("Average accuracy: {}".format(np.average(accuracies)))

    # # d2 = datetime.now()

    # # print(d2-d1)
