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
from sklearn.preprocessing import PolynomialFeatures

# Import our ML algorithms
from ML.validation import cross_validate_algo
from ML.validation import cross_validate_dm_argmax
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
import utils.data_transform as data_transform
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

    no_coord_features           = True # Keeping coords as features improves performance :/
    ensemble_testing            = False
    downsampled_param_search    = False
    downsample                  = True
    dm_test                     = False
    summarise_labels            = False
    load_query                  = True

    from utils import dm_gp_comparison; dm_gp_comparison.dm_vs_gp()

    ######## LOAD DATA ########
    print("Loading data from npzs...")
    labels, labelcounts, bath_locations, features = data.load_training_data()
    multi_locations, multi_features, multi_labels_lists = data.load_multi_label_data()
    if summarise_labels == True:
        multi_labels = data.summarised_labels(multi_labels_lists)
    multi_labels = data_transform.multi_label_counts(multi_labels_lists, zero_indexed=False)

    # Don't load full dataset without sufficient free memory
    if load_query and psutil.virtual_memory().available >= 2e9:
        qp_locations, validQueryID, x_bins, query, y_bins = data.load_test_data()

        print("Filter down to non-nan queries and locations...")
        valid_query_idxs = np.where( (~np.isnan(query).any(axis=1) & np.isfinite(query).all(axis=1)) )[0]
        query = query[valid_query_idxs]
        qp_locations = qp_locations[valid_query_idxs]
        infinite_idx = np.where(~np.isfinite(query).all(axis=1))[0]
        query_sn = scale(normalize(query))

    ######### FEATURES ##########
    print("Loading features...")
    features = np.array(features)

    # Remove long/lat coordinates
    if no_coord_features:
        features = features[:,2:]
        try:
            query_sn = query_sn[:,2:]
        except NameError:
            print("query points weren't loaded into memory")

        except NameError:
            print("query points weren't loaded into memory")

    # NOTE _s suffix kept here for clarity
    print("Scaling features...")
    features_sn = (normalize(scale(features), axis=0)) # 0.8359, 0.5323 for PoGPE
    # features_s = scale(features) # MARGINALLY better than normalize(scale)
    
    ########### DOWNSAMPLING ##########
    pf = PolynomialFeatures(2)

    if config['downsample'] == True:
        red_coords, red_features, red_mlabels, ml_argsort = data_transform.downsample(bath_locations, features_sn, multi_labels)
        f = pf.fit_transform(red_features)
        q = pf.fit_transform(query_sn)

    ######## DOWNSAMPLING PARAM SEARCH #########
    if config['downsampled_param_search'] == True:
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
    
    # labels = np.array(labels)
    labels_simple = data.summarised_labels(labels)

    if config['dm_test'] == True:
        dm = DirichletMultinomialRegression()
        print("Fitting DM regressor...")
        dm.fit(f, red_mlabels)
        print("Forming predictions...")
        preds_dm = dm.predict(q)
        # vis.show_map(qp_locations, preds_dm.argmax(axis=1), display=False, filename='full_predictions_dirmul_simplelabels_2016-09-11')
        # vis.show_map(qp_locations, preds_dm.argmax(axis=1), display=False, filename='full_predictions_dirmul_polyspace2', vmin=1, vmax=24)

        # f = pf.fit_transform(features_sn)
        f = features_sn
        q = query_sn

        # res3 = benchmarks.dm_vs_det_stats(preds_dm, preds_gp)

    ######## PLOT LR/RF ########
    # f = pf.fit_transform(features_sn)
    # f = features_sn
    # q = query_sn

    # lr = LogisticRegression()
    # lr.fit(f, labels)
    # preds_lr = lr.predict(q)
    # # res1 = benchmarks.dm_vs_det_stats(preds_dm, preds_lr)
    # vis.show_map(qp_locations, preds_lr, display=False, vmin=1, vmax=24, filename='full_predictions_logisticregression_polyspace2')

    # rf = RandomForestClassifier()
    # rf.fit(f, labels)
    # preds_rf = rf.predict(q)
    # # res2 = benchmarks.dm_vs_det_stats(preds_dm, preds_rf)
    # vis.show_map(qp_locations, preds_rf, display=False, vmin=1, vmax=24, filename='full_predictions_randomforest_nocoords')
    ###########################

    # vis.show_map(bath_locations, labels, display=False, vmin=1, vmax=24, filename='original_map_plot')

    preds_gp = np.load('data/plain_gp_simplelabels_querypreds.npy')

    # size = 100
    # train_idx = data.mini_batch_idxs(labels_simple, size, 'even')
    train_idx = np.load('data/semi-optimal-1000-subsample.npy')
    test_idx = np.array(list(set(np.arange(features.shape[0])) - set(train_idx)))

    # from ML.gp.revrand_glm import revrand_glm, RGLM
    # rglm = RGLM(nbases=2000)
    # print("fitting glm")
    # # f = pf.fit_transform(features_sn[train_idx])
    # rglm.fit(f[train_idx], labels_simple[train_idx])
    # print("predicting glm")
    # q = pf.fit_transform(f[test_idx])
    # pr = rglm.predict(f[test_idx])

    # res = cross_validate_dm_argmax(f, red_mlabels, DirichletMultinomialRegression())
    # vis.histogram(freqs, title='Full Multi-labels Histogram', filename='hist_full_multi_labels.pdf', offset=1)

    # vis.clear_plt()

    # freqs = np.concatenate((np.bincount(labels)[1:], [0]))
    # vis.histogram(freqs, title='Full Labels Histogram', filename='hist_full_labels.pdf', offset=1)

    # vis.clear_plt()

    # freqs = np.concatenate((np.bincount(labels_simple), [0]))
    # vis.histogram(freqs, title='Simplified Labels Histogram', filename='hist_simple_labels.pdf')

    # foo = np.concatenate((np.bincount(data.summarised_labels(np.concatenate(multi_labels_lists))), [0]))
    # vis.histogram(foo, title='Simplified Multi-labels Histogram', filename='hist_simple_multi_labels.pdf') 

    ########################################### Product of Experts ###########################################
    if ensemble_testing == True:
        benchmarks.GP_ensemble_tests(features_sn, labels_simple, train_idx)

    #########################################################################################################


    ############################################ Visualisation #############################################
    x_bins_training, y_bins_training = list(set(bath_locations[:,0])), list(set(bath_locations[:,1]))
    # vis.show_map(qp_locations, query[:,2], x_bins, y_bins, display=False)
    # vis.show_map(bath_locations, labels, x_bins_training, y_bins_training, display=False, vmin=1, vmax=24, filename='training_class_map')
    # vis.show_map(bath_locations, labels, x_bins_training, y_bins_training, vmin=1, vmax=24)
    #########################################################################################################

    # benchmarks.classification_dummy_testing()
