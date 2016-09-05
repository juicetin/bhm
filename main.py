#!/bin/python3
# Account for headless server/no display backend
import matplotlib as mpl
import psutil
import os
if "DISPLAY" not in os.environ: # or os.environ['DISPLAY'] == ':0':
    mpl.use('SVG')
mpl.use('SVG')

import numpy as np
import copy
import sys
import math
from datetime import datetime
import pdb
import GPy

from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
from ML.gp.gp_mt import GPMT
from ML.dir_mul.dirichlet_multinomial import DirichletMultinomialRegression
from ML.dir_mul.nicta.dirmultreg import dirmultreg_learn, dirmultreg_predict

import utils
import utils.visualisation as vis
import utils.load_data as data
import utils.benchmarks as benchmarks
import utils.gpy_benchmark as gpy_benchmarks

def info(type, value, tb):
    """
    This function is for replacing Python's default exception hook
    so that we drop into the debugger on uncaught exceptions to be
    able to immediately identify the issue/potentially recover
    operations
    """
    import traceback, pdb
    traceback.print_exception(type, value, tb)
    print
    pdb.pm()

# Main function
if __name__ == "__main__":
    sys.excepthook = info

    no_coord_features = False # Keeping coords as features improves performance :/
    ensemble_testing = False
    downsampled_param_search = False

    ######## LOAD DATA ########
    print("Loading data from npzs...")
    labels, labelcounts, bath_locations, features = data.load_training_data()
    multi_locations, multi_features, multi_labels = data.load_multi_label_data()
    # multi_labels = data.summarised_labels(multi_labels)
    multi_labels = data.multi_label_counts(multi_labels)

    ######### FEATURES ##########
    print("Loading features...")
    features = np.array(features)

    # Remove long/lat coordinates
    if no_coord_features:
        features = features[:,2:]

    # NOTE _s suffix kept here for clarity
    print("Scaling features...")
    features_sn = (normalize(scale(features), axis=0)) # 0.8359, 0.5323 for PoGPE
    # features_s = scale(features) # MARGINALLY better than normalize(scale)
    
    ########### DOWNSAMPLING ##########
    from utils.downsample import downsample_spatial_data
    print("Downsampling data...")
    red_coords, red_features, red_mlabels = downsample_spatial_data(bath_locations, features_sn, multi_labels)
    ml_argsort = np.argsort(red_mlabels.sum(axis=1))

    ######## DOWNSAMPLING PARAM SEARCH #########
    if downsampled_param_search == True:
        mlabels_max = np.argmax(red_mlabels, axis=1)
        mlabels_max_bins = np.bincount(mlabels_max)
        errors = np.zeros(red_mlabels.shape[0])
        for i in range(1, red_mlabels.shape[0]):
            print(i, end=' ', flush=True)
            W = dirmultreg_learn(red_features[ml_argsort[-i:]], red_mlabels[ml_argsort[-i:]])
            preds = dirmultreg_predict(red_features, W)[0]
            preds_max = np.argmax(preds, axis=1)
            preds_max_bins = np.bincount(preds_max)
            labels_error = np.sum(np.abs(preds_max_bins - mlabels_max_bins))
            errors[i-1] = labels_error
        print()
    
    # vis.show_map(red_coords, np.argmax(red_mlabels, axis=1), display=False)
    # vis.show_map(red_coords, np.argmax(red_mlabels, axis=1))
    # vis.show_map(red_coords, np.argmax(red_mlabels,axis=1), np.unique(red_coords[:,0]), np.unique(red_coords[:,1]), display=False, vmin=0, vmax=23, filename='reduced_training_map')
    # vis.show_map(red_coords, np.argmax(red_mlabels,axis=1), np.unique(red_coords[:,0]), np.unique(red_coords[:,1]), vmin=0, vmax=23)

    # Don't load full dataset without sufficient free memory
    if psutil.virtual_memory().available >= 2e9:
        qp_locations, validQueryID, x_bins, query, y_bins = data.load_test_data()

        print("Filter down to non-nan queries and locations...")
        valid_query_idxs = np.where( (~np.isnan(query).any(axis=1) & np.isfinite(query).all(axis=1)) )[0]
        query = query[valid_query_idxs]
        qp_locations = qp_locations[valid_query_idxs]
        infinite_idx = np.where(~np.isfinite(query).all(axis=1))[0]
        query_sn = scale(normalize(query))

    dm = DirichletMultinomialRegression()
    print("Fitting DM regressor...")
    dm.fit(red_features, red_mlabels)

    # labels = np.array(labels)
    labels_simple = data.summarised_labels(labels)

    ########################################### Product of Experts ###########################################
    if ensemble_testing == True:
        size = 100
        train_idx = data.mini_batch_idxs(labels_simple, size, 'even')
        # train_idx = np.load('data/semi-optimal-1000-subsample.npy')

        test_idx = np.array(list(set(np.arange(features.shape[0])) - set(train_idx)))

        gp = GaussianProcess()
        gp.fit(features_sn[train_idx], labels_simple[train_idx])

        gp = GaussianProcess(classification_type='OvR')
        gp_stats = benchmarks.testGP(gp, features_sn, labels_simple, train_idx, n_iter=1)
        print("normal GP: {} \n\taverages: {} {}".format( gp_stats, np.average(gp_stats[0]), np.average(gp_stats[1])))

        n_iter=1
        gp1 = PoGPE(200)
        gp1_stats = benchmarks.testGP(gp1, features_sn, labels_simple, train_idx, n_iter=n_iter)
        print("PoE: {} \n\taverages:{} {}".format(gp1_stats, np.average(gp1_stats[0]), np.average(gp1_stats[1])))

        gp11 = PoGPE(500)
        gp11_stats = benchmarks.testGP(gp11, features_sn, labels_simple, train_idx, n_iter=1)
        print("PoE: {} \n\taverages: {} {}".format( gp11_stats, np.average(gp11_stats[0]), np.average(gp11_stats[1])))

        gp12 = PoGPE(1000)
        gp12_stats = benchmarks.testGP(gp12, features_sn, labels_simple, train_idx, n_iter=1)
        print("PoE: {} \n\taverages: {} {}".format( gp12_stats, np.average(gp12_stats[0]), np.average(gp12_stats[1])))

        gp2 = GPoGPE(200)
        gp2_stats = benchmarks.testGP(gp2, features_sn, labels_simple, train_idx, n_iter=n_iter)
        print("PoGPE: {} \n\taverages: {} {}".format( gp2_stats, np.average(gp2_stats[0]), np.average(gp2_stats[1])))

        gp3 = BCM(200)
        gp3_stats = benchmarks.testGP(gp3, features_sn, labels_simple, train_idx, n_iter=n_iter)
        print("BCM: {} \n\taverages: {} {}".format( gp3_stats, np.average(gp3_stats[0]), np.average(gp3_stats[1])))

        gp4 = rBCM(200)
        gp4_stats = benchmarks.testGP(gp4, features_sn, labels_simple, train_idx, n_iter=5)
        print("BCM: {} \n\taverages: {} {}".format( gp4_stats, np.average(gp4_stats[0]), np.average(gp4_stats[1])))

        print("PoE: {} \n\taverages:{} {}".format(gp1_stats, np.average(gp1_stats[0]), np.average(gp1_stats[1])))
        print("PoGPE: {} \n\taverages: {} {}".format( gp2_stats, np.average(gp2_stats[0]), np.average(gp2_stats[1])))
        print("BCM: {} \n\taverages: {} {}".format( gp3_stats, np.average(gp3_stats[0]), np.average(gp3_stats[1])))

        # GPy benchmarking
        test_idx = np.array(list(set(np.arange(16502)) - set(train_idx)))
        preds = gpy_benchmark.gpy_bench(features_sn, labels_simple, train_idx)
        auroc = helpers.roc_auc_score_multi(labels_simple[test_idx], preds)
        score = helpers.score(labels_simple[test_idx], np.argmax(preds, axis=0))
        print(auroc, score)

    #########################################################################################################


    ############################################ Visualisation #############################################
    x_bins_training, y_bins_training = list(set(bath_locations[:,0])), list(set(bath_locations[:,1]))
    # vis.show_map(qp_locations, query[:,2], x_bins, y_bins, display=False)
    # vis.show_map(bath_locations, labels, x_bins_training, y_bins_training, display=False, vmin=1, vmax=24, filename='training_class_map')
    # vis.show_map(bath_locations, labels, x_bins_training, y_bins_training, vmin=1, vmax=24)
    #########################################################################################################

    # benchmarks.classification_dummy_testing()
