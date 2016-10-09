import numpy as np
import pdb
from utils import visualisation as vis
from ML.dir_mul.nicta.dirmultreg import dirmultreg_learn, dirmultreg_predict
from ML.gp.gp_gpy import GPyC

from ML.validation import cross_validate_algo
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import progressbar

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
    algos = [LogisticRegression(), SVC(), KNeighborsClassifier(), RandomForestClassifier()]
    results = ""
    for algo in algos:
        print('Now calculating {}'.format(str(algo).split('(')[0]))
        results += cross_validate_algo(features, labels, 10, algo)

    return results
