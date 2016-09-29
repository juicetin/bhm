from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from utils.downsample import downsample_spatial_data

def poly_features(data, polyspace=2):
    pf = PolynomialFeatures(polyspace)
    return pf.fit_transform(data)

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

def multi_label_counts(labels, zero_indexed=False):
    """
    Converts lists of uneven category counts into a bincount of them in a uniform matrix 
    """

    uniqs = np.unique(np.concatenate(labels, axis=0))
    num_uniqs = uniqs.shape[0]
    multi_labels = np.array([fill(labellist, num_uniqs, zero_indexed) for labellist in labels])
    multi_labels = np.concatenate(multi_labels, axis=0).reshape(labels.shape[0], num_uniqs)

    return multi_labels

def downsample(locations, features, labels, method='fixed-gried'):
    print("Downsampling data...")
    red_coords, red_features, red_mlabels = downsample_spatial_data(locations, features, labels, method=method)
    ml_argsort = np.argsort(red_mlabels.sum(axis=1))
    return red_coords, red_features, red_mlabels, ml_argsort

def summarise_list(old_lst, label_map):
    new_lst = np.copy(old_lst)
    for k, v in label_map.items():
        new_lst[old_lst == k] = v
    return new_lst

def summarised_labels(labels):

    label_map={1:0,2:0,3:1,4:3,5:1,6:3,7:3,8:3,9:3,10:1,11:3,12:3,13:2,14:2,15:2,16:1,17:1,18:0,19:1,20:0,21:0,22:1,23:0,24:0}

    if isinstance(labels[0], np.ndarray):
        print('Summarising multi label list...')

        multi_labels = np.zeros((labels.shape[0], 4), dtype=np.int64)
        for orig_l, summarised_l in label_map.items():
            multi_labels[:,summarised_l] += labels[:,orig_l-1]
        return multi_labels

        # return np.array([summarise_list(np.array(labels[i]), label_map) for i in range(labels.shape[0])])

    else:
        print('Summarising label list...')
        return summarise_list(labels, label_map)
        # for k, v in label_map.items(): 
        #     new_labels[labels==k] = v

