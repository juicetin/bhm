import numpy as np
import pdb
from utils import visualisation as vis
from utils import load_data 
from ML.dir_mul import dm_mcmc
from ML.gp.gp_gpy import GPyC
from ML.gp import gp_multi_gpy as gpym
from ML.gp.poe import PoGPE
from ML.gp.gpoe import GPoGPE
from ML.gp.bcm import BCM
from ML.gp.rbcm import rBCM
from ML import helpers
from datetime import datetime

from ML import validation
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import multiprocessing as mp
from multiprocessing import Pool
import itertools

from utils import visualisation as vis
from utils import data_transform
from ML.dir_mul.nicta.dirmultreg import dirmultreg_learn, dirmultreg_predict, predict_parallel
from ML import pseudo_multioutput
from utils import downsample

from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from utils.load_data import inverse_indices

import progressbar
import itertools
import pymc
import time

def timing(func):
    def wrapper(*arg, **kw):
        '''source: http://www.daniweb.com/code/snippet368.html'''
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        return (t2 - t1), res, func.__name__
    return wrapper

def algo_module_to_str(algo):
    return str(algo()).split('(')[0]

def search_even_split_areas(q_preds, q_vars):
    """
    Wrapper for find_even_split_areas to search many parameters for the largest matching areas
    """
    labels = np.arange(q_preds.shape[1])
    bounds_list = np.arange(.02, .36, .02)

    # Iterate over all label pairs #TODO more than just pairs though?
    for label_pair in itertools.combinations(labels, 2):

        # Compare labels with the following occupancy bounds
        for cur_bounds in itertools.combinations(itertools.combinations(bounds_list, 2), 2):

            # Get co-habitation stats
            _, _, idxs = find_even_split_areas(q_preds, q_vars, bounds=cur_bounds, split_labels=label_pair)

            # See if variance is lower than on average
            split_var_avg = np.average(q_vars[idxs])
            rest_var_avg = np.average(q_vars[inverse_indices(q_preds, idxs)])
            if split_var_avg < rest_var_avg:
                print(label_pair, cur_bounds)
                print(split_var_avg, rest_var_avg)

            print('Average variance in area: {}, number of points: {}'.format(split_var_avg, idxs[0].shape[0]))

def find_even_split_areas(q_preds, q_vars, bounds=[[0.1, 0.4], [0.3, 0.9]], split_labels=[1,2], check='preds'):
    """
    Finds areas which contain certain mixes of labels with a minimum threshold of equality
    """
    if check == 'preds':
        q_comp = q_preds
    elif check == 'vars':
        q_comp = q_vars
    else:
        raise ValueError("Must check either 'preds' or 'vars', please provide a valid option.")

    even_split_idxs = np.where((q_comp[:,split_labels[0]] > bounds[0][0]) & 
                               (q_comp[:,split_labels[0]] < bounds[0][1]) & 
                               (q_comp[:,split_labels[1]] > bounds[1][0]) & 
                               (q_comp[:,split_labels[1]] < bounds[1][1]))
    even_splits_preds = q_preds[even_split_idxs]
    even_splits_vars = q_vars[even_split_idxs]
    
    avg_var = np.average(even_splits_vars)
    print('Average variance in condition-satisfied splits was: {}'.format(avg_var))
    print('There were {} matches'.format(even_splits_preds.shape[0]))
    return even_splits_preds, even_splits_vars, even_split_idxs

def check_overlaps(f, l, q, reg=100):
    W = dirmultreg_learn(f, l, verbose=True, reg=reg)
    query_preds = dirmultreg_predict(q, W)
    return find_even_split_areas(query_preds)

def f_max_var_rank(f, v):
    """
    Returns the n-rank of the largest matching labels' variance
    """
    f_max, f_max_idx = f.max(), f.argmax()
    f_max_var = v[f_max_idx]
    rank = np.argwhere(v == f_max_var)[0][0]
    return rank

def plot_training_data_per_label(locations, labels, gen_images=False):
    """
    Plots all points per label on separate maps to visualise where they are/be able to semi-manually find images of a certain label
    """
    uniq_labels = np.unique(labels)
    file_base = 'training_points_'+str(uniq_labels.shape[0])+'labels_'
    label_coord_map = {}
    for c in uniq_labels:
        cur_label_idx = np.where(labels == c)
        cur_label_locations = locations[cur_label_idx]
        cur_labels = labels[cur_label_idx]
        if gen_images == True:
            vis.show_map(cur_label_locations, cur_labels, display=False, filename=file_base+str(c))
        label_coord_map[c] = cur_label_locations

    return label_coord_map

def dm_vs_gp_even_split_vars(dm_results, gp_results):
    """
    Compares the predictions/variance of a DM vs GP in areas where they believe there are even splits
    """
    dm_preds, _, dm_vars = dm_results
    gp_preds, gp_vars = gp_results

    # Check vars in GP compared to DM for even splits
    dm_evensplit_preds, dm_evensplit_vars = find_even_split_areas(dm_preds, dm_vars)
    gp_evensplit_preds, gp_evensplit_vars = find_even_split_areas(gp_preds, gp_vars)

def GP_preds_score(features, labels, train_idx, test_idx):
    gp = GPyC()
    gp.fit(features[train_idx], labels[train_idx])
    preds = gp.predict(features[test_idx])
    acc = np.sum(preds[0].argmax(axis=0) == labels[test_idx])/test_idx.shape[0]
    print(acc)

def chain_stats(chains, features, labels):
    """
    Goes through a list of weight chains, and finds the one/s with the least variance
    after forming predictions using them and using the results
    """
    dm_errs = np.zeros(chains.shape[0])
    dm_vars = np.zeros(chains.shape[0])
    bar = progressbar.ProgressBar(maxval=chains.shape[0])
    bar.start()
    for i, weights in enumerate(chains):
        bar.update(i)
        dm_stats = dirmultreg_predict(features, weights.reshape(chains.shape[1], chains.shape[2]))
        dm_errs[i] = np.average(np.abs(dm_stats[0] - labels))
        dm_vars[i] = np.average(dm_stats[2])
    bar.finish()
    
    err_min, err_min_idx = dm_errs.min(), dm_errs.argmin()
    vars_min, vars_min_idx = dm_vars.min(), dm_vars.argmin()

    lowest_err_varidx = list(dm_errs.argsort()).index(vars_min_idx)
    lowest_var_erridx = list(dm_vars.argsort()).index(err_min_idx)
    print("The weights with lowest error are also ranked {} in terms of lowest variance".format(lowest_err_varidx))
    print("The lowest variance's error is also ranked {} in terms of lowest error".format(lowest_var_erridx))

    return dm_errs, dm_vars

def dm_vs_gp_matching(dm_preds, dm_vars, gp_preds, gp_vars, even_split_idxs):
    # dm_preds[even_split_idxs]
    # gp_preds[even_split_idxs]

    dm_vars[even_split_idxs]
    gp_vars[even_split_idxs]

def algo_module_to_str(algo):
    return str(algo).split('.')[-1][:-2]

def det_scores(features, labels_sets):
    algos = [LogisticRegression, SVC, KNeighborsClassifier, RandomForestClassifier]
    results = ""
    for label_set in labels_sets:
        for algo in algos:
            print('Now calculating {}'.format(algo_module_to_str(algo)))
            results += validation.cross_validate_algo(features, label_set, 10, algo)
    return results

def det_multi_scores(features, label_sets):
    algos = [LinearRegression, SVR, KNeighborsRegressor, RandomForestRegressor]
    results = "Algorithm & Average Error & Labels Used & Average Row Sum* & Min \\\\\\hline"
    for label_set in label_sets:
        for algo in algos:
            print('Now calculating {} for {}-labels'.format(algo_module_to_str(algo), label_set.shape[1]))
            results += validation.cross_validate_algo_multioutput(features, label_set, algo, folds=10)
    return results

def dm_scores_all_sets(features, label_sets):
    feature_sets = [
        features,
        features[:,2:],
        data_transform.features_squared_only(features),
        data_transform.features_squared_only(features[:,2:]),
        PolynomialFeatures().fit_transform(features),
        PolynomialFeatures().fit_transform(features[:,2:])
    ]

    return dm_scores(feature_sets, label_sets)

def dm_scores(feature_sets, label_sets):
    """
    Prints a latex table of results of possible combinations between a list of 
    features and list of DM multi-labels
    """

    results ="Features & Root Mean Squared Error \\\\\\hline\n"
    for i, (features, labels) in enumerate(itertools.product(feature_sets, label_sets)):
        print('Now calculating label set: {}'.format(i))
        results += validation.cross_validate_dm(features, labels, 10)

    return results

def check_dm_err_var_rankings(dm_mc_errs, dm_mc_vars):
    errs_argsort = dm_mc_errs.argsort()
    vars_argsort = dm_mc_vars.argsort()

    for i, idx in enumerate(vars_argsort):
        print('{}-smallest variance corresponds to the {}-smallest error - index {}'.format(i, np.where(errs_argsort == idx)[0][0], idx))

def plot_det_maps(query_features, qp_locations, all_preds=None, features=None, labels=None):
    if all_preds == None:
        print('Making det predictions')
        print('SVC predictions...')
        svc_preds = SVC().fit(features, labels).predict(query_features)
        print('Logistic Regression predictions...')
        lr_preds = LogisticRegression().fit(features, labels).predict(query_features)
        print('kNN predictions')
        knn_preds = KNeighborsClassifier().fit(features, labels).predict(query_features)
        print('Random Forest predictions...')
        rf_preds = RandomForestClassifier().fit(features, labels).predict(query_features)
    else:
        lr_preds, svc_preds, knn_preds, rf_preds = all_preds
    
    det4_preds = np.column_stack((svc_preds, lr_preds, knn_preds, rf_preds))

    vis.plot_multi_maps(qp_locations, det4_preds, filename='det4_preds', title_list=['SVM', 'Logistic Regression', 'kNN', 'Random Forest'])

def plot_det_multi_maps(qp_locations, preds):
    lr4_preds, svm4_preds, knn4_preds, rf4_preds = preds
    vis.plot_multi_maps(qp_locations, lr4_preds, filename='multi4_preds_lr', offset=0)
    vis.plot_multi_maps(qp_locations, svm4_preds, filename='multi4_preds_svm', offset=0)
    vis.plot_multi_maps(qp_locations, knn4_preds, filename='multi4_preds_knn', offset=0)
    vis.plot_multi_maps(qp_locations, rf4_preds, filename='multi4_preds_rf', offset=0)

def multi_dm_mcmc_chains(features, labels, iters=2000000):
    # dirmultreg_learn(features, labels, activation='soft', reg=100, verbose=False, iters=2000000)

    nprocs = mp.cpu_count() - 1
    jobs = range(nprocs)
    args = [(features, labels, 'soft', 100, False, iters, i) for i in jobs]
    pool = Pool(processes=nprocs)
    print("Distributing MCMC sampling across {} processes...".format(nprocs))
    parallel_mcmc_chains_models = pool.starmap(dm_mcmc.dirmultreg_learn, args)

    return np.array(parallel_mcmc_chains_models)

def multi_dm_mcmc_chains_continue(features, labels, iters=100000):
    nprocs = mp.cpu_count() - 1
    jobs = range(nprocs)
    args = [(features, labels, 'soft', 100, False, iters, i) for i in jobs]
    pool = Pool(processes=nprocs)
    print("Distributing MCMC sampling across {} processes...".format(nprocs))
    parallel_mcmc_chains_models = pool.starmap(dm_mcmc.continue_mcmc, args)
    # return np.array(parallel_mcmc_chains_models)

def save_dm_mcmc(*, l):
    chain_sizes = {4:int(9e6), 24:int(9.5e5)}
    size_strs = {4:'9m', 24:'950k'}

    nprocs = mp.cpu_count() - 1
    for i in range(nprocs):
        print('Saving chain from {}...'.format('mcmc_db/dm{}_mcmc_{}.pickle'.format(l, i)))
        db = pymc.database.pickle.load('mcmc_db/dm{}_mcmc_{}.pickle'.format(l, i))
        np.save('mcmc_db/dm{}_mcmc_{}'.format(l, i), db.trace('mean', chain=None)[:][:chain_sizes[l]])
        del(db)

    # db0 = pymc.database.pickle.load('mcmc_db/dm{}_mcmc_{}.pickle'.format(l, 0))
    # db1 = pymc.database.pickle.load('mcmc_db/dm{}_mcmc_{}.pickle'.format(l, 1))
    # db2 = pymc.database.pickle.load('mcmc_db/dm{}_mcmc_{}.pickle'.format(l, 2))

    # chains.append(db.trace('mean', chain=None)[:][:chain_sizes[l]])

def plot_dm_hists_per_chain_4l():
    print('Loading memory mapped data...')
    dm4_0 = np.load('mcmc_db/dm4_mcmc_9m_0.npy', mmap_mode='r').reshape(9000000, 4*19)
    dm4_1 = np.load('mcmc_db/dm4_mcmc_9m_1.npy', mmap_mode='r').reshape(9000000, 4*19)
    dm4_2 = np.load('mcmc_db/dm4_mcmc_9m_2.npy', mmap_mode='r').reshape(9000000, 4*19)

    print('Plotting hists of 0...')
    vis.plot_dm_hists(dm4_0[4500000::10], filename='dm4_9m_0_mcmc_weight_hist')
    print('Plotting hists of 1...')
    vis.plot_dm_hists(dm4_1[4500000::10], filename='dm4_9m_1_mcmc_weight_hist')
    print('Plotting hists of 2...')
    vis.plot_dm_hists(dm4_2[4500000::10], filename='dm4_9m_2_mcmc_weight_hist')

def plot_dm_hists_per_chain_24l():
    print('Loading memory mapped data...')
    dm24_0 = np.load('mcmc_db/dm24_mcmc_950k_0.npy', mmap_mode='r').reshape(950000, 24*19)
    dm24_1 = np.load('mcmc_db/dm24_mcmc_950k_1.npy', mmap_mode='r').reshape(950000, 24*19)

    print('Plotting hists of 0...')
    vis.plot_dm_hists(dm24_0, filename='dm24_950k_0_mcmc_weight_hist')
    print('Plotting hists of 1...')
    vis.plot_dm_hists(dm24_1, filename='dm24_950k_1_mcmc_weight_hist')

def test_dm_data(features, labels):
    f_sq2 = data_transform.features_squared_only(features)

    f = scale(normalize(f_sq2))
    W = dirmultreg_learn(f, labels)
    p = dirmultreg_predict(f, W)
    avg_err = np.average(np.abs(p[0] - labels))
    print('scale(normalize()): {}'.format(avg_err))

    f = scale(normalize(f_sq2), axis=1)
    W = dirmultreg_learn(f, labels)
    p = dirmultreg_predict(f, W)
    avg_err = np.average(np.abs(p[0] - labels))
    print('scale(normalize(), axis=1): {}'.format(avg_err))

def plot_map_with_variance_threshold(locations, predictions, variances, var_threshold):
    print(var_threshold)
    idxs = np.where(variances < var_threshold)[0]
    vis.plot_multi_maps(locations[idxs], predictions[idxs], offset=0, 
            filename='{}l-preds-{}var_limit'.format(predictions.shape[1], var_threshold))

def dm_maps_from_chains(*, chains, coords, features, argmax=True):
    if argmax==True:
        plot_map_func = vis.show_map
    elif chains.shape[1] == 4:
        plot_map_func = vis.plot_multi_maps
    elif chains.shape[1] == 24:
        plot_map_func = vis.plot_dm_per_label_maps_multi # (q_locations, q_preds, filename='dm_alllabels_heatmap')
    for i, chain in enumerate(chains):
        print('creating {}-th map'.format(i))
        preds = predict_parallel(features, chain)[0]
        if argmax == True:
            plot_map_func(coords, preds.argmax(axis=1), filename='dm{}_argmax_images/dm_heatmap_{}'.format(preds.shape[1], i))
        else:
            plot_map_func(coords, preds, filename='dm{}_images/dm_heatmap_{}'.format(preds.shape[1], i))

def calc_all_det_preds(features, l4, l24, query):
    algos = [LogisticRegression, SVC, KNeighborsClassifier, RandomForestClassifier]
    preds = []
    for algo in algos:
        preds4 = algo().fit(features, l4).predict(query)
        preds24 = algo().fit(features, l24).predict(query)
        preds.append(preds4)
        preds.append(preds24)

    return preds

def save_all_det_preds(preds):
    lr4p, lr24p, svm4p, svm24p, knn4p, knn24p, rf4p, rf24p = preds
    np.save('data/lr4p', lr4p)
    np.save('data/lr24p', lr24p)
    np.save('data/svm4p', svm4p)
    np.save('data/svm24p', svm24p)
    np.save('data/knn4p', knn4p)
    np.save('data/knn24p', knn24p)
    np.save('data/rf4p', rf4p)
    np.save('data/rf24p', rf24p)

def calc_all_det_multi_preds(features, l4, l24, query):
    algos = [LinearRegression, SVR, KNeighborsRegressor, RandomForestRegressor]
    preds = []
    for algo in algos:
        print('Currently doing multi-output preds for {} with 4 labels...'.format(algo_module_to_str(algo)))
        preds4 = pseudo_multioutput.predict(features, query, l4, algo)
        print('Now 24...')
        preds24 = pseudo_multioutput.predict(features, query, l24, algo)
        preds.append(preds4)
        preds.append(preds24)

    return preds

def save_all_det_multi_preds(multi_preds):
    lr4mp, lr24mp, svm4mp, svm24mp, knn4mp, knn24mp, rf4mp, rf24mp = multi_preds
    np.save('data/lr4mp', lr4mp)
    np.save('data/lr24mp', lr24mp)
    np.save('data/svm4mp', svm4mp)
    np.save('data/svm24mp', svm24mp)
    np.save('data/knn4mp', knn4mp)
    np.save('data/knn24mp', knn24mp)
    np.save('data/rf4mp', rf4mp)
    np.save('data/rf24mp', rf24mp)

# def calc_gp_preds(features, l4, l24, query, ret=False, parallel=False, save=True):
#     print('4-labels, single-label')
#     gp = GPyC()
#     gp.fit(features, l4, parallel=True)
#     np.save('data/gp4_p_models', gp.models)
#     gp_preds = gp.predict(query, parallel=parallel)
#     del(gp)
#     if save == True:
#         np.save('data/gp4_p', gp_preds)
# 
#     gp_preds = gp.predict(query, parallel=parallel)
#     del(gp)
#     print('24-labels, single-label')
#     gp = GPyC()
#     gp.fit(features, l24, parallel=True)
#     np.save('data/gp24_p_models', gp.models)
#     gp_preds = gp.predict(query, parallel=parallel)
#     del(gp)
# 
#     if save == True:
#         np.save('data/gp24_p', gp_preds)

def calc_gp_multi_preds(features, l4, l24, query, parallel=False, ret=False, gp_true=None, save=True):
    print('4-labels, multi-label')
    gp = gpym.GPyMultiOutput()
    gp.fit(features, l4, parallel=True)
    np.save('data/gp4_mp_models', gp.models)
    gp_preds = np.array(gp.predict(query, parallel=parallel))
    if parallel==True:
        gp_preds = np.array([np.concatenate(gp_preds[::2]), np.concatenate(gp_preds[1::2])])
        # gp_preds = np.concatenate(gp_preds, axis=2)
        # gp_preds = np.array([np.concatenate(gp_preds[:3]), np.concatenate(gp_preds[3:])])
    del(gp)
    if save == True:
        np.save('data/gp4_mp', gp_preds)

    print('24-labels, multi-label')
    gp = gpym.GPyMultiOutput()
    gp.fit(features, l24)
    np.save('data/gp24_mp_models', gp.models)
    gp_preds = gp.predict(query)
    del(gp)
    
    if save == True:
        np.save('data/gp24_mp', gp_preds)

def downsample_queries(qp_locs, queries):
    qp_red_coords, qp_red_features, _, qp_red_idxs = downsample.downsample_spatial_data(qp_locs, queries, np.ones(queries.shape[0]), 'fixed-grid')
    np.save('data/qp_red_coords'   ,qp_red_coords)
    np.save('data/qp_red_features' ,qp_red_features)
    np.save('data/qp_red_idxs'     ,qp_red_idxs)

# def search_contiguous_confident_splits(preds):
#     for label_pair in itertools.combinations(range(preds.shape[1]), 2):
#         _, _, idxs = find_even_split_areas(q_preds, q_vars, bounds=[[0.1, 0.4], [0.3, 0.9]], split_labels=label_pair, check='preds')
#         # vis.plot_multi_maps(coords, preds[0][idxs][:,label_pair],

def test_naive_GP_time(train_features, train_labels, test_features=None, parallel=False, predict=True, optimize=True):
    print('Fit/predict in parallel: {}, doing predictions: {}'.format(parallel, predict))

    gp = GPyC()
    t1 = datetime.now()
    gp.fit(train_features, train_labels, parallel=parallel, optimize=optimize)
    t2 = datetime.now()
    print('time taken to train on all training features: {}, with {} parallel and {} model optimisation'.format(t2-t1, parallel, optimize))

    if predict == True:
        gp4_p = gp.predict(test_features, parallel)
        t3 = datetime.now()
        print('time taken to predict all query points: {}'.format(t3-t2))
        np.save('preds/gp4_p_fixed_{}'.format(train_features.shape[0]), gp4_p)

    return gp

def test_naive_GP_time_combo(train_features, train_labels, test_features=None, predict=True, parallel=False, optimize=True, start=1):
    for i in np.arange(start, 6):
        idx = i*1000
        print('Now testing GP runtime for {} points'.format(idx))
        test_naive_GP_time(train_features[:idx], train_labels[:idx], test_features, parallel=parallel, predict=predict, optimize=optimize)

def plot_entropy(coords, entropies, threshold=-1e-3):
    idxs = helpers.discard_outlier_entropies(entropies, threshold)
    print('{}% of points discarded'.format((1-idxs.shape[0]/coords.shape[0])*100))
    norm_entr = helpers.normalise_entropies(entropies[idxs])
    vis.show_map(coords[idxs], norm_entr, filename='entropy_map')
    return norm_entr

def map_many_entropy_thresholds(coords, entropies, *, labels):
    for threshold in np.arange(10, 20000, 10):
        idxs = helpers.discard_outlier_entropies(entropies, threshold)
        vis.show_map(coords[idxs], helpers.normalise_entropies(entropies[idxs]), filename='dm{}_entropies_t{}'.format(labels, threshold))

def final_entropy_maps(coords, entr4, entr24):
    new_entr4 = helpers.tune_entropies_better_spread(entr4, 200, rungs=5, stepsize=100)
    vis.show_map(coords, new_entr4, filename='dm4_entropy_map')

@timing
def biodiversity_searching(preds, coords, plot=False):
    """
    Determine a basic/probably ecologically invalid definition biodiverse clusters 
    based on the very simple criteria that (n) number of points simulatneously occur
    at a rate greater than some threshold (t) that decreases as the number of simulatneous
    occurrences required increases (as we can't have 24 labels all occuring more than 10% of
    the time, for example...)
    """
    t1 = datetime.now()
    scale_factor=.8
    prev_threshold = (1/scale_factor) * 0.15
    for i in np.arange(2, preds.shape[1]):
        cur_threshold = scale_factor * prev_threshold
        prev_threshold = cur_threshold
        cur_cohab_count = i
        cur_cohab_idxs = np.where(np.sum(preds > cur_threshold, axis=1) > i)[0]
        print('There are [{}] points where [{}] labels occur more than [{}] of the time'.format(
            cur_cohab_idxs.shape[0], cur_cohab_count, cur_threshold))

        # Plot the map displaying the cohabitated areas at the current location
        if plot == True:
            vis.show_map(coords[cur_cohab_idxs], np.zeros(cur_cohab_idxs.shape[0]), 
                filename='dm24_cohab_map_{}habs_{}points'.format(cur_cohab_count, cur_cohab_idxs.shape[0]),
                title='Cohabitation between {} habitats, with a {} occurrence threshold - {} points'.format(
                    i, np.round(cur_threshold, 3), cur_cohab_idxs.shape[0]))
    print(datetime.now()-t1)

def biodiversity_for_cohab_count(preds, cohabitations=2, factor=1.2):
    """
    Don't use this!
    """
    print("don't use this!")
    labels = np.arange(0, preds.shape[1])
    label_sets = itertools.combinations(labels, cohabitations)
    cur_cohabitat_idxs = set()
    for pair in label_sets:
        for axis in pair:
            cur_cohabitat_idxs.union(np.where(preds[:,axis] > (1/cohabitations*factor))[0])
            # [np.where(preds[:,axis] > (1/cohabitations*factor)) for axis in pair]

    return cur_cohabitat_idxs

def dm_toyplots(Xtrc=None, Xtec=None, Xtr=None, Xte=None, Ctr=None, Cte=None, size=1000):
    if Xtrc == None and Xtec == None and Xtr == None and Xte == None and Ctr == None and Cte == None:
        Xtrc, Xtec, Xtr, Xte, Ctr, Cte = load_data.generate_dm_toy_ex(True, False, size)
    sz = int(Xtrc.shape[0]/3)
    # print('Each cluster test size is {}'.format(sz))
    
    # vis.plot_multi_maps(Xtrc, Ctr/Ctr.sum(axis=1)[:,np.newaxis], offset=0, filename='toyheatmaps', across=2, down=1)
    Xcs = np.concatenate((Xtrc, Xtec))
    Cs = np.concatenate((Ctr, Cte))
    Cs_norm = Cs/Cs.sum(axis=1)[:,np.newaxis]

    vis.scatter_toymap_clusters(Xcs, filename='toy_scattermap_clusters.pdf')
    vis.scatter_multi_maps(Xcs, Cs_norm, filename='toy_scattermap')

    vis.scatter_multi_maps(Xtec, Cte, filename='toy_scattermap_test_expected')

    W = dirmultreg_learn(data_transform.features_squared_only(Xtr), Ctr, reg=100)
    dm_preds = dirmultreg_predict(data_transform.features_squared_only(Xte), W)
    dm_errs = np.abs(dm_preds[0] - Cte/Cte.sum(axis=1)[:,np.newaxis])

    print('DM avg errs: {}, clusterA: {}, clusterB: {}, clusterC: {}'.format( 
        np.average(dm_errs), np.average(dm_errs[:sz]), np.average(dm_errs[sz:2*sz]), np.average(dm_errs[2*sz:])))
    vis.scatter_multi_maps(Xtec, dm_preds[0], filename='toy_scattermap_dmpreds')

    gp = GPyC()
    gp.fit(Xtr, Ctr.argmax(axis=1), parallel=True)
    gp_preds = gp.predict(Xte, parallel=True)
    gp_errs = gp_preds[0].argmax(axis=1) == Cte.argmax(axis=1)
    print('GP avg acc: {}, clusterA: {}, clusterB: {}, clusterC: {}'.format( 
        np.average(gp_errs), np.average(gp_errs[:sz]), np.average(gp_errs[sz:2*sz]), np.average(gp_errs[2*sz:])))
    vis.scatter_map(Xtec, Cte.argmax(axis=1), filename='toy_scattermap_testgp_expected')
    vis.scatter_map(Xtec, gp_preds[0].argmax(axis=1), filename='toy_scattermap_gppreds')

    return dm_preds, gp_preds, Xtrc, Xtec, Xtr, Xte, Ctr, Cte

def plot_toydata_vars(dm_preds, gp_preds):
    sz = dm_preds[0].shape[0]/3
    dm_entr = dm_preds[3]
    gp_vars = np.average(gp_preds[1], axis=1)
    dm_entr_stacked = np.column_stack((dm_entr[:sz], dm_entr[sz:2*sz], dm_entr[2*sz:]))
    gp_vars_stacked = np.column_stack((gp_vars[:sz], gp_vars[sz:2*sz], gp_vars[2*sz:]))

    vis.plot_multiple_arrays(dm_entr_stacked, filename='toy_scattermap_dm_entropy.pdf', datatype='entropy')
    vis.plot_multiple_arrays(gp_vars_stacked, filename='toy_scattermap_gp_vars.pdf', datatype='variance')

def plot_dm_argmax_from_chains(coords, features, chains, i_offset=0):
    for i, W in enumerate(chains):
        dm_preds = predict_parallel(features, W, 2)
        if chains.shape[1] == 4:
            vis.show_map(coords, dm_preds[0].argmax(axis=1), filename='dm4_argmax_images/dm4_argmax_{}'.format(i+i_offset))
        elif chains.shape[1] == 24:
            vis.show_map(coords, dm_preds[0].argmax(axis=1), filename='dm24_argmax_images/dm24_argmax_{}'.format(i+i_offset), vmin=1, vmax=14)

def plot_gp_vars(coords, gp_preds):
    gp_p, gp_v = gp_preds
    max_idxs = gp_p.argmax(axis=1)

    gp_max_vars = (gp_v[np.arange(gp_v.shape[0])[:,np.newaxis], max_idxs[:,np.newaxis]]).flatten()
    gp_max_probs = (gp_p[np.arange(gp_p.shape[0])[:,np.newaxis], max_idxs[:,np.newaxis]]).flatten()

    print('Average across all variances was {}, but average across the variances corresponding to the argmax labels was {}'.format(
        np.average(gp_v), np.average(gp_max_vars) ))

    vis.show_map(coords, np.sqrt(gp_max_vars), filename='gp4_max_st', save_im=False)
    vis.show_map(coords, np.average(gp_v, axis=1), filename='gp4_avg_vars', save_im=False)
    vis.show_map(coords, gp_max_probs, filename='gp4_max_probs', save_im=False)
    # vis.standalone_colorbar(im1, 'gp4_max_vars')
    # vis.standalone_colorbar(im2, 'gp4_avg_vars')
    # vis.standalone_colorbar(im3, 'gp4_max_probs')
