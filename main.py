#!/bin/python3
# Account for headless server/no display backend import matplotlib as mpl import psutil import os
import os
import matplotlib as mpl
import psutil
if "DISPLAY" not in os.environ: # or os.environ['DISPLAY'] == ':0':
    mpl.use('SVG')
mpl.use('Agg')

from os.path import join, dirname
from dotenv import load_dotenv, find_dotenv

from importlib import reload
import numpy as np
import copy
import sys
import math
from datetime import datetime
import pdb
import GPy
from ML.gp.gp_gpy import GPyC
import ML.gp.gp_multi_gpy as gpm

from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn.metrics import roc_auc_score
from sklearn import neighbors
from sklearn.preprocessing import PolynomialFeatures

# Import our ML algorithms
from ML.validation import cross_validate_algo
from ML.validation import cross_validate_dm_argmax
from ML.validation import cross_validate_dm
from ML.knn import kNN
from ML.random_forests import rf
from ML.logistic_regression import lr
from ML import pseudo_multioutput

from ML import helpers
from ML.gp.gp import GaussianProcess
from ML.gp.poe import PoGPE
from ML.gp.gpoe import GPoGPE
from ML.gp.bcm import BCM
from ML.gp.rbcm import rBCM
from ML.gp.gp_mt import GPMT
from ML.dir_mul.dirichlet_multinomial import DirichletMultinomialRegression
from ML.dir_mul.nicta.dirmultreg import dirmultreg_learn, dirmultreg_predict
from ML.dir_mul.dm_mcmc import dirmultreg_learn as dm_mcmc_learn
from ML import validation as val

from ML.dir_mul import dm_mcmc

import utils
from utils import visualisation as vis
from utils import load_data as data
from utils import data_transform as data_transform
import utils.benchmarks as benchmarks
import utils.gpy_benchmark as gpy_benchmarks
from utils import downsample
from utils import thesis_experiments


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

    # Load environment variables
    load_dotenv(find_dotenv())

    config = {}
    # config['no_coord_features']          = False # NOTE no longer need this, load preprocessed directly. don't change this!!! Keeping coords as features improves performance :/
    # config['ensemble_testing']           = False
    # config['downsampled_param_search']   = False
    # config['downsample']                 = False # No need for this, loading from disk-cached versions now
    # config['dm_test']                    = False
    # config['summarise_labels']           = False
    # config['load_query']                 = True

    config['no_coord_features']          = int(os.environ.get('no_coord_features'))
    config['ensemble_testing']           = int(os.environ.get('ensemble_testing'))
    config['downsampled_param_search']   = int(os.environ.get('downsampled_param_search'))
    config['downsample']                 = int(os.environ.get('downsample'))
    config['dm_test']                    = int(os.environ.get('dm_test'))
    config['summarise_labels']           = int(os.environ.get('summarise_labels'))
    config['load_query']                 = int(os.environ.get('load_query'))

    # props = data.load_squidle_data()  
    # zip_obj = zip(props['latitude'], props['longitude'])  
    # utm_coords = data.latlong_to_utm(zip_obj) 

    EC, C_test_norm, gp_preds, gp_vars, X_train_c, X_test_c, X_train, X_test, C_train, C_test = data.load_dm_vs_gp_pickles()
    # vis.dm_pred_vs_actual(EC, C_test_norm, filename='toy_dm_pred_plot')
    # vis.gp_pred_vs_actual(C_test_norm, gp_preds, gp_vars, display=False, filename='gp_with_vars')

    # from utils import dm_gp_comparison; dm_gp_comparison.dm_vs_gp()

    ######## LOAD DATA ########
    print("Loading data from npzs...")
    labels, labelcounts, bath_locations, features = data.load_training_data()
    _, _, bath_locations, _ = data.load_training_data()
    # multi_locations, multi_features, multi_labels_lists = data.load_multi_label_data()
    # multi_labels = data_transform.multi_label_counts(multi_labels_lists, zero_indexed=False)
    if config['summarise_labels'] == True:
        multi_labels = data_transform.summarised_labels(multi_labels)
    # multi_labels_norm = multi_labels/multi_labels.sum(axis=1)[:,np.newaxis]
    
    red_features, red_mlabels4, red_mlabels24, red_coords, red_scoords, \
        red_sfeatures, red_slabels4, red_slabels24 = data.load_reduced_data()

    f_sq2 = data_transform.features_squared_only(red_features)
    l4 = red_mlabels4
    l4_norm = l4/l4.sum(axis=1)[:,np.newaxis]
    l24 = red_mlabels24
    l24_norm = l24/l24.sum(axis=1)[:,np.newaxis]
    W4 = np.load('data/W4.npy')
    W24 = np.load('data/W_2m_1444288.npy')

    # Don't load full dataset without sufficient free memory
    if config['load_query'] == True and psutil.virtual_memory().available >= 2e9:
        print('Loading query data')
        qp_locations, query_sn = data.load_fixed_query_data()

        # qp_locations, validQueryID, x_bins, query, y_bins = data.load_test_data()

        # print("Filter down to non-nan queries and locations...")
        # # valid_query_idxs = np.where( (~np.isnan(query).any(axis=1) & np.isfinite(query).all(axis=1)) )[0]
        # valid_query_idxs = np.load('data/valid_query_idxs.npy')
        # query = query[valid_query_idxs]
        # qp_locations = qp_locations[valid_query_idxs]
        # # infinite_idx = np.where(~np.isfinite(query).all(axis=1))[0]
        # query_sn = scale(normalize(query))

    ######### FEATURES ##########
    # print("Loading features...")
    # features = np.array(features)

    # Remove long/lat coordinates
    if config['no_coord_features'] == True:
        features = features[:,2:]
        try:
            query_sn = query_sn[:,2:]
        except NameError:
            print("query points weren't loaded into memory")

    # f_pf2 = data_transform.features_squared_only(features)

    # NOTE _s suffix kept here for clarity
    # print("Scaling features...")
    # features_sn = (normalize(scale(features), axis=1)) # 0.8359, 0.5323 for PoGPE
    # # features_s = scale(features) # MARGINALLY better than normalize(scale)
    # # labels = np.array(labels)
    # labels_simple = data_transform.summarised_labels(labels)
    
    ########### DOWNSAMPLING ##########
    # pf = PolynomialFeatures(2)

    if config['downsample'] == True:
        # red_coords, red_features, red_mlabels, ml_argsort = data_transform.downsample(bath_locations, features_sn, multi_labels, method='fixed-grid')
        red_coords, red_features, red_mlabels, ml_argsort = data_transform.downsample(bath_locations, features_sn, multi_labels, method='cluster-rules')
        f = pf.fit_transform(red_features)

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

    if config['load_query'] == True:
        q_sq2 = np.load('data/q_sq2.npy')
        # q = query_sn
        # q_sq2 = data_transform.features_squared_only(query_sn)
        # res3 = benchmarks.dm_vs_det_stats(preds_dm, preds_gp)


    # dm4_preds = dirmultreg_predict(q_sq2, W4)
    # dm24_preds = dirmultreg_predict(q_sq2, W24)

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

    preds_gp = np.load('data/plain_gp_simplelabels_querypreds.npy', mmap_mode='r')

    # size = 100
    # train_idx = data.mini_batch_idxs(labels_simple, size, 'even')
    train_idx = np.load('data/semi-optimal-1000-subsample.npy')
    test_idx = np.array(list(set(np.arange(red_features.shape[0])) - set(train_idx)))

    ########################################### Product of Experts ###########################################
    if config['ensemble_testing'] == True:
        benchmarks.GP_ensemble_tests(red_features, labels_simple, train_idx)

    #########################################################################################################


    ############################################ Visualisation #############################################
    x_bins_training, y_bins_training = list(set(bath_locations[:,0])), list(set(bath_locations[:,1]))
    # vis.show_map(qp_locations, query[:,2], x_bins, y_bins, display=False)
    # vis.show_map(bath_locations, labels, x_bins_training, y_bins_training, display=False, vmin=1, vmax=24, filename='training_class_map')
    # vis.show_map(bath_locations, labels, x_bins_training, y_bins_training, vmin=1, vmax=24)
    #########################################################################################################


    ######### Looking at error and variance of DM using different projections #########
    # all_errs, all_vars = benchmarks.dm_test_feature_space(red_features, l_norm)

    # chains = dm_mcmc.dirmultreg_learn(f_sq2, l_norm)

    # # Load dm mc 4-label chain stats - vars, errs
    # dmmc4_errs = np.load('data/dmmc4_errs.npy')
    # dmmc4_vars = np.load('data/dmmc4_vars.npy')
    # dm4chains = np.load('data/dm_mcmc_30000_4l.npy', mmap_mode='r') #13362

    # dm_mc_errs24 = np.load('data/dm_mc_errs24.npy', mmap_mode='r')
    # dm_mc_vars24 = np.load('data/dm_mc_vars24.npy', mmap_mode='r')


    # # Get 'common split' areas
    # if dmmc4_errs.argmin() != dmmc4_vars.argmin():
    #     raise ValueError('dmmc4 error and variance argmin should match!')
    # W = dm4chains[dmmc4_errs.argmin()]
    # dm_q_preds = dirmultreg_predict(q_sq2, W)
    # cmp_axes = [1,3]
    # dm_es_preds, dm_es_vars, dm_es_idxs = thesis_experiments.find_even_split_areas(dm_q_preds[0], dm_q_preds[2], bounds=[0.2, 0.4], split_labels=cmp_axes)

    # # Generate padded predictions to fill original map area
    # dm_es_preds_padded = np.empty(dm_q_preds[0].shape)
    # dm_es_preds_padded[:] = None
    # dm_es_preds_padded[dm_es_idxs[0]] = dm_es_preds

    # # Load GP stats
    # gp_q_preds = np.load('data/gp_query_preds.npy')
    # gp_q_vars = np.load('data/gp_query_vars.npy')

    # # Create GP stats in areas of DM even split coords
    # gp_dmes_preds = gp_q_preds.T[dm_es_idxs[0]]
    # gp_dmes_vars = gp_q_vars.T[dm_es_idxs[0]]

    # vis.plot_dm_per_label_maps(qp_locations, dm_es_preds_padded[:,cmp_axes]) 

    # thesis_experiments.det_maps(f_sq2, red_mlabels4, q_sq2)

