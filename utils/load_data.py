import numpy as np
from pandas import read_csv
import pdb; import code
import utm

import matplotlib.pyplot as plt
from revrand import basis_functions as bases
from ML.dir_mul.nicta.dirmultreg import dirmultreg_learn, dirmultreg_predict # originally yavanna
from scipy.misc import logsumexp # originally yavanna

import utils.visualisation as vis

# Load all the data
def load_training_data():
    bath_and_dom_lowres = np.load('data/bathAndDomLabel_lowres.npz')
    label = bath_and_dom_lowres['label']
    labelcounts = bath_and_dom_lowres['labelcounts']
    bath_locations = bath_and_dom_lowres['locations']
    features = bath_and_dom_lowres['features']
    
    return (label, labelcounts, bath_locations, features)

def load_multi_label_data():
    multi_label_data = np.load('data/multiple_labels_training_data.npz')
    locations = multi_label_data['locations']
    labels = multi_label_data['labels']
    features = multi_label_data['features']    
    return locations, features, labels

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

def csv_to_npz(filename):
    data = read_csv(filename, sep=',')
    return data

def fill(labels, num_uniqs, zero_indexed=False):
    counts = np.bincount(labels)
    if zero_indexed == False:
        missing = num_uniqs - len(counts) + 1
    else:
        missing = num_uniqs - len(counts)
    if missing > 0:
        counts = np.concatenate((counts, [0] * missing), axis=0)

    if zero_indexed != False:
        return list(counts)
    else:
        return list(counts)[1:]

def generate_dm_toy_ex(plot_toy_graph=False, plot_cluster_distr=False):
    # Settings
    size = 100
    multisamp1 = 500
    multisamp2 = 10
    multisamp3 = 50
    kfolds = 5
    activation = 'soft'

    # Make data
    X1 = np.random.multivariate_normal([-5, -5], [[1, 0], [0, 1]], size)
    X2 = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], size)
    X3 = np.random.multivariate_normal([-5, 5], [[1, 0], [0, 1]], size)
    C1 = np.random.multinomial(multisamp1, [0.7, 0.3], size)
    C2 = np.random.multinomial(multisamp2, [0.3, 0.7], size)
    C3 = np.random.multinomial(multisamp3, [0.5, 0.5], size)

    # Concatenate data
    X_train_coords = np.vstack((X1[:size/2], X2[:size/2], X3[:size/2]))
    C_train = np.vstack((C1[:size/2], C2[:size/2], C3[:size/2]))
    X_test_coords = np.vstack((X1[size/2:], X2[size/2:], X3[size/2:]))
    C_test = np.vstack((C1[size/2:], C2[size/2:], C3[size/2:]))
    X_train = bases.RadialBasis(X_train_coords).transform(X_train_coords, lenscale=1)
    X_test = bases.RadialBasis(X_test_coords).transform(X_test_coords, lenscale=1)

    # X_train_coords = np.vstack((X1, X2))
    # C_train = np.vstack((C1, C2))
    # X_test_coords = np.vstack((X3))
    # C_test = np.vstack((C3))
    # X_train = bases.RadialBasis(X1).transform(X_train_coords, lenscale=1)
    # X_test = bases.RadialBasis(X_test_coords).transform(X_test_coords, lenscale=1)

    if plot_toy_graph == True:
        vis.plot_toy_data(np.concatenate((X1, X2, X3)), np.concatenate((C_train, C_test)).argmax(axis=1), filename='toydataplot.pdf', display=False)
    if plot_cluster_distr == True:
        vis.plot_multilabel_distribution(C1/C1.sum(axis=1)[:,np.newaxis], title='cluster A distribution', filename='toy_clusterA_distr.pdf', display=False)
        vis.plot_multilabel_distribution(C2/C2.sum(axis=1)[:,np.newaxis], title='cluster B distribution', filename='toy_clusterB_distr.pdf', display=False)
        vis.plot_multilabel_distribution(C3/C3.sum(axis=1)[:,np.newaxis], title='cluster C distribution', filename='toy_clusterC_distr.pdf', display=False)

    return X_train_coords, X_test_coords, X_train, X_test, C_train, C_test

def save_dm_vs_gp_pickles():
    np.save('tmp_pickles/EC.npy', EC)
    np.save('tmp_pickles/C_test_norm.npy', C_test_norm)
    np.save('tmp_pickles/gp_preds.npy', gp_preds)
    np.save('tmp_pickles/gp_vars.npy', gp_vars)
    np.save('tmp_pickles/X_train_c.npy', X_train_c)
    np.save('tmp_pickles/X_test_c.npy', X_test_c)
    np.save('tmp_pickles/X_train.npy', X_train)
    np.save('tmp_pickles/X_test.npy', X_test)
    np.save('tmp_pickles/C_train.npy', C_train)
    np.save('tmp_pickles/C_test.npy', C_test)

def load_dm_vs_gp_pickles():
    try:
        EC          = np.load('tmp_pickles/EC.npy')
        C_test_norm = np.load('tmp_pickles/C_test_norm.npy')
        gp_preds    = np.load('tmp_pickles/gp_preds.npy')
        gp_vars     = np.load('tmp_pickles/gp_vars.npy')
        X_train_c   = np.load('tmp_pickles/X_train_c.npy')
        X_test_c    = np.load('tmp_pickles/X_test_c.npy')
        X_train     = np.load('tmp_pickles/X_train.npy')
        X_test      = np.load('tmp_pickles/X_test.npy')
        C_train     = np.load('tmp_pickles/C_train.npy')
        C_test      = np.load('tmp_pickles/C_test.npy')
        return EC, C_test_norm, gp_preds, gp_vars, X_train_c, X_test_c, X_train, X_test, C_train, C_test     
    except FileNotFoundError:
        print('Your pickles do not exist')

def utm_to_latlong(bath_locations):
    ZONE_NUMBER = 51
    ZONE_LETTER = 'S'
    return np.array([utm.to_latlon(x, y, ZONE_NUMBER, northern=False) for x, y in bath_locations])

def latlong_to_utm(lat, longs):
    """
    latlong
    """
    return np.array([utm.from_latlon(x, y)[:2] for x, y in zip(lat, longs)])

def load_squidle_data(path='../bhm-large-data/'):
    csvs = ['images-scottreef2011-2016-09-16.csv']
    keys = ['image__id', 'date_time', 'depth', 'web_location', 'latitude',
                   'longitude']

    # Set up dict
    props = {}
    for key in keys:
        props[key] = np.array([])

    # Go over all files
    for csv in csvs:
        cur_props = read_csv(path+csv)
        for key in keys:
            props[key] = np.concatenate((props[key], cur_props[key]))

    return props

# def find_matching(ll_utm, bath_locs): 

def sample_equal_multi_labels(labels):
    np.sum(multi_labels, axis=0).min()
    np.bincount(multi_labels.argmax(axis=1)).min()
