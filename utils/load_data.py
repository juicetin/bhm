import numpy as np
from pandas import read_csv
import pdb

import matplotlib.pyplot as plt
from revrand import basis_functions as bases
from ML.dir_mul.nicta.dirmultreg import dirmultreg_learn, dirmultreg_predict # originally yavanna
from scipy.misc import logsumexp # originally yavanna

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

def summarise_list(old_lst, label_map):
    new_lst = np.copy(old_lst)
    for k, v in label_map.items():
        new_lst[old_lst == k] = v
    return new_lst

def summarised_labels(labels):

    label_map={1:0,2:0,3:1,4:3,5:1,6:3,7:3,8:3,9:3,10:1,11:3,12:3,13:2,14:2,15:2,16:1,17:1,18:0,19:1,20:0,21:0,22:1,23:0,24:0}

    if isinstance(labels[0], list):
        # for i, lst in enumerate(new_labels):
        #     summarise_list(np.array(lst), np.array(labels[i]), label_map)
        return np.array([summarise_list(np.array(labels[i]), label_map) for i in range(labels.shape[0])])
    else:
        return summarise_list(labels, label_map)
        # for k, v in label_map.items(): 
        #     new_labels[labels==k] = v

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

def generate_dm_toy_ex():
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

    return X_train_coords, X_test_coords, X_train, X_test, C_train, C_test
