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

import misc.visualisation as vis
import misc.load_data as data
import misc.benchmarks as benchmarks

# Main function
if __name__ == "__main__":
    # Regression testing
    # test,pred=benchmarks.regression_dummy_testing()

    print("Loading data from npzs...")
    labels, labelcounts, bath_locations, features = data.load_training_data()
    qp_locations, validQueryID, x_bins, query, y_bins = data.load_test_data()

    print("Filter down to non-nan queries and locations...")
    valid_query_idxs = np.where( (~np.isnan(query).any(axis=1) & np.isfinite(query).all(axis=1)) )[0]
    query = query[valid_query_idxs]
    qp_locations = qp_locations[valid_query_idxs]
    infinite_idx = np.where(~np.isfinite(query).all(axis=1))[0]

    print("Loading features...")
    features = np.array(features)

    # Remove long/lat coordinates
    # features = features[:,2:]

    # NOTE _s suffix kept here for clarity
    print("Scaling features...")
    # features_s = scale(features)
    features_sn = scale(normalize(features))
    query_sn = scale(normalize(query))

    # labels = np.array(labels)
    labels_simple = data.summarised_labels(labels)

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
    # train_idx = data.mini_batch_idxs(labels_simple, size, 'even')
    train_idx = np.load('data/semi-optimal-1000-subsample.npy')

    test_idx = np.array(list(set(np.arange(features.shape[0])) - set(train_idx)))

    gp = GaussianProcess(classification_type='OvR')
    # gp.fit(features_sn[train_idx], labels_simple[train_idx])
    # y_preds, y_vars = gp.predict(features_sn[test_idx], keep_probs=True)
    # y_preds = gp.predict(features_sn[test_idx])
    gp_stats = benchmarks.testGP(gp, features_sn, labels_simple, train_idx, n_iter=2)
    print("normal GP: {} \n\taverages: {} {}".format( gp_stats, np.average(gp_stats[0]), np.average(gp_stats[1])))

    # gp1 = PoGPE(50)
    # gp1_stats = benchmarks.testGP(gp1, features_sn, labels_simple, train_idx, n_iter=1)
    # print("PoE: {} \n\taverages:{} {}".format(gp1_stats, np.average(gp1_stats[0]), np.average(gp1_stats[1])))

    # gp11 = PoGPE(500)
    # gp11_stats = benchmarks.testGP(gp11, features_sn, labels_simple, train_idx, n_iter=1)
    # print("PoE: {} \n\taverages: {} {}".format( gp11_stats, np.average(gp11_stats[0]), np.average(gp11_stats[1])))

    # gp12 = PoGPE(1000)
    # gp12_stats = benchmarks.testGP(gp12, features_sn, labels_simple, train_idx, n_iter=1)
    # print("PoE: {} \n\taverages: {} {}".format( gp12_stats, np.average(gp12_stats[0]), np.average(gp12_stats[1])))

    # gp2 = GPoGPE(200)
    # gp2_stats = benchmarks.testGP(gp2, features_sn, labels_simple, train_idx, n_iter=5)
    # print("PoGPE: {} \n\taverages: {} {}".format( gp2_stats, np.average(gp2_stats[0]), np.average(gp2_stats[1])))

    # gp3 = BCM(200)
    # gp3_stats = benchmarks.testGP(gp3, features_sn, labels_simple, train_idx, n_iter=50)
    # print("BCM: {} \n\taverages: {} {}".format( gp3_stats, np.average(gp3_stats[0]), np.average(gp3_stats[1])))

    # gp4 = rBCM(200)
    # gp4_stats = benchmarks.testGP(gp4, features_sn, labels_simple, train_idx, n_iter=5)
    # print("BCM: {} \n\taverages: {} {}".format( gp4_stats, np.average(gp4_stats[0]), np.average(gp4_stats[1])))


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
    # idx = data.mini_batch_idxs(labels_simple, size, 'even')
    # rem_idx = np.array(list(set(np.arange(16502)) - set(idx)))
    # gp = GaussianProcess()
    # gp.fit_class(features_sn[idx], labels_simple[idx])
    # preds = gp.predict_class(features_sn[rem_idx], keep_probs=True, parallel=True)
    # score = np.average(gp.roc_auc_score_multi(labels_simple[rem_idx], preds))

    # best_score = 0
    # for i in range(20):
    #     size = 1000
    #     idx = data.mini_batch_idxs(labels_simple, size, 'even')
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
    # # idx = data.mini_batch_idxs(labels_simple, size, 'even')
    # # s10 = classification_bathy_testing(features_s[idx], labels_simple[idx])
    # # s11 = classification_bathy_testing(features_sn[idx], labels_simple[idx])
    # print("For even split")
    # s1s = []
    # for i in range(100):
    #     print("Round {} for even".format(i))
    #     idx = data.mini_batch_idxs(labels_simple, size, 'even')
    #     s1s.append(benchmarks.classification_bathy_testing(features_sn[idx], labels_simple[idx]))

    # # idx = data.mini_batch_idxs(labels_simple, size, 'stratified')
    # # s2 = benchmarks.classification_bathy_testing(features_sn[idx], labels_simple[idx])
    # print("For stratified split")
    # s2s = []
    # for i in range(100):
    #     print("Round {} for stratified".format(i))
    #     idx = data.mini_batch_idxs(labels_simple, size, 'stratified')
    #     s2s.append(benchmarks.classification_bathy_testing(features_sn[idx], labels_simple[idx]))

    # print(np.average(s1s))
    # print(np.average(s2s))
    # #########################################################################################################

    # for feature_set in feature_perms:
    #     f1 = benchmarks.classification_bathy_testing(feature_set[idx], labels_simple[idx])
    #     f1s.append(f1)

    # idx = stratified_micro_batch(features, labels, 1000)
    # for feature_set in feature_perms:
    #     f1 = benchmarks.classification_bathy_testing(feature_set[idx], labels[idx])
    #     f1s.append(f1)

    # benchmarks.classification_bathy_testing(features, labels_simple, 200)


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

    # benchmarks.classification_dummy_testing()
