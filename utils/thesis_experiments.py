import numpy as np
import pdb
from utils import visualisation as vis
# from ML.dir_mul.nicta.dirmultreg import dirmultreg_learn, dirmultreg_predict
from ML.dir_mul import dm_mcmc
from ML.gp.gp_gpy import GPyC

from ML import validation
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import multiprocessing as mp
from multiprocessing import Pool

from utils import visualisation as vis

import progressbar
import itertools

# def search_even_split_areas(q_preds, q_vars):
#     """
#     Wrapper for find_even_split_areas to search many parameters for the largest matching areas
#     """
#     labels = np.arange(qp_preds.shape[1])
#     for label_pair in itertools.combinations(labels, 2):
#         for cur_bounds in ?:
#             _, split_vars, split_idxs = find_even_split_areas(q_preds, q_vars, bounds=cur_bounds, split_labels=label_pair)
#             print('Average variance in area: {}, number of points: {}'.format(np.average(split_vars), split_idxs.shape[0]))
#             del(split_vars); del(split_idxs)

def find_even_split_areas(q_preds, q_vars, bounds=[0.4, 0.6], split_labels=[1,2]):
    """
    Finds areas which contain certain mixes of labels with a minimum threshold of equality
    """
    even_split_idxs = np.where((q_preds[:,split_labels[0]] > bounds[0]) & 
                               (q_preds[:,split_labels[0]] < bounds[1]) & 
                               (q_preds[:,split_labels[1]] > bounds[0]) & 
                               (q_preds[:,split_labels[1]] < bounds[1]))
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


def det_scores(features, labels_sets):
    algos = [LogisticRegression, SVC, KNeighborsClassifier, RandomForestClassifier]
    results = ""
    for label_set in labels_sets:
        for algo in algos:
            print('Now calculating {}'.format( str(algo).split('.')[-1][:-2] ))
            results += validation.cross_validate_algo(features, label_set, 10, algo)
    return results

def check_dm_err_var_rankings(dm_mc_errs, dm_mc_vars):
    errs_argsort = dm_mc_errs.argsort()
    vars_argsort = dm_mc_vars.argsort()

    for i, idx in enumerate(vars_argsort):
        print('{}-smallest variance corresponds to the {}-smallest error - index {}'.format(i, np.where(errs_argsort == idx)[0][0], idx))

def det_maps(features, labels, query_features, qp_locations):
    print('Making det predictions')
    print('SVC predictions...')
    svc_preds = SVC().fit(features, labels).predict(query_features)
    print('Logistic Regression predictions...')
    lr_preds = LogisticRegression().fit(features, labels).predict(query_features)
    print('kNN predictions')
    knn_preds = KNeighborsClassifier().fit(features, labels).predict(query_features)
    print('Random Forest predictions...')
    rf_preds = RandomForestClassifier().fit(features, labels).predict(query_features)
    
    det4_preds = np.column_stack((svc_preds, lr_preds, knn_preds, rf_preds))

    vis.plot_multi_maps(qp_locations, det4_preds, filename='det_preds', title_list=['SVM', 'Logistic Regression', 'kNN', 'Random Forest'])

def multi_dm_mcmc_chains(features, labels, iters=2000000):
    # dirmultreg_learn(features, labels, activation='soft', reg=100, verbose=False, iters=2000000)

    nprocs = mp.cpu_count() - 1
    jobs = range(nprocs)
    args = [(features, labels, 'soft', 100, False, iters) for i in jobs]
    pool = Pool(processes=nprocs)
    print("Distributing MCMC sampling across {} processes...".format(nprocs))
    parallel_mcmc_chains_models = pool.starmap(dm_mcmc.dirmultreg_learn, args)

    return np.array(parallel_mcmc_chains_models)
