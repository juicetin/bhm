#!/bin/python3

import numpy as np
import copy
import sys
import math
from datetime import datetime
import GPy

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
from sklearn.metrics import roc_auc_score
from sklearn import neighbors

# Import our ML algorithms
from ML.validation import cross_validate_algo
from ML.knn import kNN
from ML.random_forests import rf
from ML.logistic_regression import lr

from ML import helpers
from ML.gp.gp import GaussianProcess
from ML.gp.poe import PoGPE
from ML.gp.gpoe import GPoGPE
from ML.gp.bcm import BCM
from ML.gp.rbcm import rBCM

import visualisation as vis
import gpy_benchmark

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

def mini_batch_idxs(labels, point_count, split_type):

    # Match ratios of original labels
    if split_type == 'stratified':
        sss = StratifiedShuffleSplit(labels, 1, test_size=point_count/len(labels))
        for train_index, test_index in sss:
            pass
        return test_index

    # Have even number of labels ignoring original ratios
    elif split_type == 'even':

        # Find how many of each class to generate
        uniq_classes = np.unique(labels)
        num_classes = len(uniq_classes)
        class_size = int(point_count/num_classes)
        class_sizes = np.full(num_classes, class_size, dtype='int64')

        # Adjust for non-divisiblity
        rem = point_count % class_size
        if rem != 0:
            class_sizes[-1] += rem

        # Generate even class index list
        class_idxs = np.concatenate(np.array(
            [np.random.choice(np.where(labels==cur_class)[0], cur_class_size, replace=False)
                for cur_class, cur_class_size 
                in zip(uniq_classes, class_sizes)
            ]
        ), axis=0)

        return class_idxs

        # 

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
    # cv = StratifiedShuffleSplit(labels, 1, test_size=0.1)
    scores = []
    for train_index, test_index in cv:
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        gp.fit_class(X_train, y_train)
        y_pred = gp.predict_class(X_test, keep_probs=True)
        # return y_pred

        score = gp.roc_auc_score_multi(y_test, y_pred)
        # score = roc_auc_score(y_test, y_pred)
        scores.append(score)
        print("Average AUROC score for this round is: {}".format(score))

    avg_score = np.average(scores)
    print("Average AUROC score: {}".format(avg_score))
    return avg_score

def regression_dummy_testing():
    ####################################################################################
    # Dummy testing - regression
    ####################################################################################

    X, y = datasets.make_regression(n_samples=500, n_features=3)
    print(type(y[0]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    gp = GaussianProcess()
    gp.fit(X_train, y_train)
    y_pred, variances = gp.predict(X_test)
    mse = helpers.regression_score(y_test, y_pred)
    print(y_test)
    print(y_pred)
    print(mse)

    return y_test, y_pred

def classification_dummy_testing():
    ####################################################################################
    # Dummy testing - classification
    ####################################################################################

    gp = GaussianProcess()

    #X, y = datasets.make_circles(n_samples=12)
    # n_informative
    # n_redundant
    # n_repeated
    redundant = 0
    repeated = 0
    classes = 2
    clusters = 2
    informative = math.ceil(math.sqrt(classes*clusters))
    features = informative + repeated + redundant

    # classes * clusters <= 2**informative
    # informative + redundant + repeated < features
    X, y = datasets.make_classification(n_samples=200,
            n_features=features, 
            n_clusters_per_class=clusters,
            n_redundant=redundant, 
            n_repeated=repeated,
            n_informative=informative,
            n_classes=classes)

    from sklearn.cross_validation import StratifiedKFold
    # cv = StratifiedKFold(y, n_folds = 20)
    cv = StratifiedShuffleSplit(y, 1, test_size=20/len(X))
    accuracies = []
    for train_index, test_index in cv:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # LR
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = gp.score(y_pred, y_test)
        print("LR F-score is: {}".format(accuracy))
        vis.plot_classes(X_test, y_test, y_pred)

        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = gp.score(y_pred, y_test)
        print("RF F-score is: {}".format(accuracy))
        vis.plot_classes(X_test, y_test, y_pred)

        clf = SVC()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = gp.score(y_pred, y_test)
        print("SVM F-score is: {}".format(accuracy))
        vis.plot_classes(X_test, y_test, y_pred)

        # GP
        gp.fit_class(X_train, y_train)
        y_pred = gp.predict_class(X_test)
        accuracy = gp.score(y_pred, y_test)
        accuracies.append(accuracy)
        print("GP F-score is: {}".format(accuracy))
        vis.plot_classes(X_test, y_test, y_pred)
    # print("Average accuracy: {}".format(np.average(accuracies)))

def testGP(gp, features, labels, idx, n_iter=5):
    rem_idx = np.array(list(set(np.arange(features.shape[0])) - set(idx)))
    aurocs = []
    scores = []
    for i in range(n_iter):
        gp.fit(features[idx], labels[idx])
        means, var = gp.predict(features[rem_idx], keep_probs=True)
        auroc = helpers.roc_auc_score_multi(labels[rem_idx], means)
        score = helpers.score(labels[rem_idx], np.argmax(means, axis=0))
        print("This round: auroc: {}, f1: {}".format(auroc, score))
        aurocs.append(auroc)
        scores.append(score)
    return aurocs, scores


# Main function
if __name__ == "__main__":
    test,pred=regression_dummy_testing()
    sys.exit(0)

    # # print("Loading data from npzs...")
    # print("Loading data from npzs...")
    # labels, labelcounts, bath_locations, features = load_training_data()
    # qp_locations, validQueryID, x_bins, query, y_bins = load_test_data()

    # print("Filter down to non-nan queries and locations...")
    # valid_query_idxs = np.where( (~np.isnan(query).any(axis=1) & np.isfinite(query).all(axis=1)) )[0]
    # query = query[valid_query_idxs]
    # qp_locations = qp_locations[valid_query_idxs]
    # infinite_idx = np.where(~np.isfinite(query).all(axis=1))[0]

    # print("Loading features...")
    # features = np.array(features)
    # # Remove long/lat coordinates
    # # features = features[:,2:]

    # # NOTE _s suffix kept here for clarity
    # print("Scaling features...")
    # # features_s = scale(features)
    # features_sn = scale(normalize(features))
    # query_sn = scale(normalize(query))

    # # labels = np.array(labels)
    # labels_simple = summarised_labels(labels)

    # NOTE best for simple classes - scaling, then normalising features
    # order: original, normalised, scaled, normalised-scaled, scaled-normalised
    # # [0.13911784653850162,   0.62549115824070989,  0.80419877283841434,  0.80067072027980024, 0.77420703810759661, 
    # #  0.0014678899082568807, 0.017750714162373553, 0.012602890116646892, 0.02482014086796169, 0.029004894912527762]
    # feature_perms = [features, features_n, features_s, features_ns, features_sn]

    # [0.74880756856008601, 0.67981634815766179, 0.70133102810814296]
    # [0.72739502569246961, 0.71183799527284519, 0.65632134135368103]
    # feature_perms = [features_s, features_ns, features_sn]

    ########################################### Product of Experts ###########################################

    # size = 100
    # train_idx = mini_batch_idxs(labels_simple, size, 'even')

    train_idx = np.load('data/semi-optimal-1000-subsample.npy')

    # gp = GaussianProcess()
    # gp_stats = testGP(gp, features_sn, labels_simple, train_idx, n_iter=2)
    # gp1 = PoGPE(50)
    # gp1_stats = testGP(gp1, features_sn, labels_simple, train_idx, n_iter=1)
    # gp11 = PoGPE(500)
    # gp11_stats = testGP(gp11, features_sn, labels_simple, train_idx, n_iter=1)
    # gp12 = PoGPE(1000)
    # gp12_stats = testGP(gp12, features_sn, labels_simple, train_idx, n_iter=1)
    # gp2 = GPoGPE(200)
    # gp2_stats = testGP(gp2, features_sn, labels_simple, train_idx, n_iter=5)

    # gp3 = BCM(200)
    # gp3_stats = testGP(gp3, features_sn, labels_simple, train_idx, n_iter=50)
    # print("BCM: {} {} {}", gp3_stats, np.average(gp3_stats[0]), np.average(gp3_stats[1]))

    # gp4 = rBCM(200)
    # gp4_stats = testGP(gp4, features_sn, labels_simple, train_idx, n_iter=5)
    # print("BCM: {} {} {}", gp4_stats, np.average(gp4_stats[0]), np.average(gp4_stats[1]))

    # print("normal GP: {} {} {}", gp_stats, np.average(gp_stats[0]), np.average(gp_stats[1]))
    # print("PoE: {} {} {}", gp1_stats, np.average(gp1_stats[0]), np.average(gp1_stats[1]))
    # print("PoE: {} {} {}", gp11_stats, np.average(gp11_stats[0]), np.average(gp11_stats[1]))
    # print("PoE: {} {} {}", gp12_stats, np.average(gp12_stats[0]), np.average(gp12_stats[1]))
    # print("PoGPE: {} {} {}", gp2_stats, np.average(gp2_stats[0]), np.average(gp2_stats[1]))

    # GPy benchmarking
    # test_idx = np.array(list(set(np.arange(16502)) - set(train_idx)))
    # preds = gpy_benchmark.gpy_bench(features_sn, labels_simple, train_idx)
    # auroc = helpers.roc_auc_score_multi(labels_simple[test_idx], preds)
    # score = helpers.score(labels_simple[test_idx], np.argmax(preds, axis=0))
    # print(auroc, score)

    #########################################################################################################

    ########################################### Actual prediction ###########################################
    # size = 200
    # # idx = np.load('data/semi-optimal-500-subsample.npy')
    # idx = mini_batch_idxs(labels_simple, size, 'even')
    # rem_idx = np.array(list(set(np.arange(16502)) - set(idx)))
    # gp = GaussianProcess()
    # gp.fit_class(features_sn[idx], labels_simple[idx])
    # preds = gp.predict_class(features_sn[rem_idx], keep_probs=True, parallel=True)
    # score = np.average(gp.roc_auc_score_multi(labels_simple[rem_idx], preds))

    # best_score = 0
    # for i in range(20):
    #     size = 1000
    #     idx = mini_batch_idxs(labels_simple, size, 'even')
    #     rem_idx = np.array(list(set(np.arange(16502)) - set(idx)))
    #     gp = GaussianProcess()
    #     gp.fit_class(features_sn[idx], labels_simple[idx])
    #     # predictions = np.full(len(query_sn), -1, dtype='int64')
    #     preds = gp.predict_class(features_sn[rem_idx], keep_probs=True, parallel=True)
    #     score = np.average(gp.roc_auc_score_multi(labels_simple[rem_idx], preds))
    #     print("Average AUROC for this round: {}".format(score))

    #     if score > best_score:
    #         best_score = score
    #         best_idxs = idx

    # Incrementally predict. TODO embarassingly parallelisable
    # for i in range(0, len(query_sn), 1000):
    #     print("Current start: {}".format(i))
    #     start = i
    #     end = start+1000
    #     if (end > len(query_sn)):
    #         end = len(query_sn)
    #     iter_preds = gp.predict_class(query_sn[start:end])
    #     predictions[start:end] = iter_preds
      
    # gp.predict_class(query_sn)
    #########################################################################################################

    #################################### Test performance on bathy data  ####################################
    # size = 500
    # # idx = mini_batch_idxs(labels_simple, size, 'even')
    # # s10 = classification_bathy_testing(features_s[idx], labels_simple[idx])
    # # s11 = classification_bathy_testing(features_sn[idx], labels_simple[idx])
    # print("For even split")
    # s1s = []
    # for i in range(100):
    #     print("Round {} for even".format(i))
    #     idx = mini_batch_idxs(labels_simple, size, 'even')
    #     s1s.append(classification_bathy_testing(features_sn[idx], labels_simple[idx]))

    # # idx = mini_batch_idxs(labels_simple, size, 'stratified')
    # # s2 = classification_bathy_testing(features_sn[idx], labels_simple[idx])
    # print("For stratified split")
    # s2s = []
    # for i in range(100):
    #     print("Round {} for stratified".format(i))
    #     idx = mini_batch_idxs(labels_simple, size, 'stratified')
    #     s2s.append(classification_bathy_testing(features_sn[idx], labels_simple[idx]))

    # print(np.average(s1s))
    # print(np.average(s2s))
    #########################################################################################################

    # for feature_set in feature_perms:
    #     f1 = classification_bathy_testing(feature_set[idx], labels_simple[idx])
    #     f1s.append(f1)

    # idx = stratified_micro_batch(features, labels, 1000)
    # for feature_set in feature_perms:
    #     f1 = classification_bathy_testing(feature_set[idx], labels[idx])
    #     f1s.append(f1)

    # classification_bathy_testing(features, labels_simple, 200)


    ########################################## Compare other Algos ###########################################
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

    # 10-fold cross-validation for all
    # results = []
    # for classifier in classifiers:
    #     results.append(cross_validate_algo(features, labels, 10, classifier))

    # # 10-fold cross-validation for all
    # for classifier in classifiers:
    #     cross_validate_algo(features, labels, 10, classifier)
    #########################################################################################################

    ############################################ Visualisation #############################################
    # x_bins_training, y_bins_training = list(set(bath_locations[:,0])), list(set(bath_locations[:,1]))
    # vis.show_map(qp_locations, query[:,2], x_bins, y_bins, display=False)
    # vis.show_map(bath_locations, labels, x_bins_training, y_bins_training, display=False)
    #########################################################################################################

    # classification_dummy_testing()
    pass
