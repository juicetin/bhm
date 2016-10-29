from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from utils.downsample import downsample_spatial_data
import pdb

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
    """
    Summarise labels based on mapping provided by marine experts.
    """

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

def features_squared_only(features):
    """
    Square features (not full quadratic expansion), and include the 1 bias term.
    """
    return np.concatenate((np.hstack((features, features**2)), np.ones(features.shape[0])[:,np.newaxis]), 1)

def merge_rare_labels(labels, min_count=20):
    """
    As some labels will be very low in occurrence as a result of simplifying the multi-label counts down to
    their argmaxes (in some cases, labels may occur 0 times), it may be beneficial to merge them for a
    number of reasons, one being that cross-fold validation can then be done, as labels that occur, say, 2
    times, cannot be split up into several folds.
    """
    new_labels = np.copy(labels)
    label_map={1:0,2:0,3:1,4:3,5:1,6:3,7:3,8:3,9:3,10:1,11:3,12:3,13:2,14:2,15:2,16:1,17:1,18:0,19:1,20:0,21:0,22:1,23:0,24:0}

    # Build reverse label map
    simplelabels = np.unique(list(label_map.values()))
    reverse_label_map = {}
    for label in simplelabels:
        reverse_label_map[label] = []
    for key in label_map:
        reverse_label_map[label_map[key]].append(key)
    print(reverse_label_map)

    bins = np.bincount(labels)
    rare_idxs = np.where(bins < min_count)[0]
    print('The rare labels are: {}'.format(rare_idxs))

    # Merge rare labels with the most common one corresponding to their parent label
    for i in rare_idxs:
        parent_label = label_map[i+1]
        max_count = 0
        sibling_counts = np.array([(label, bins[label-1]) for label in reverse_label_map[parent_label]])
        substitute_label = sibling_counts[:,0][sibling_counts[:,1].argmax()]
        print('Rare label {} is being substituted with {}'.format(i, substitute_label))
        new_labels = np.where(new_labels == i, substitute_label, new_labels)

    return new_labels

def scale_dm_preds(preds, lower_bound=0.2, upper_bound=0.4):
    """
    Scales predictions if too many of the values sit within too small a range. E.g. for dm24 predictions, 96% of the values reside between 0.0 and 0.2, while the remaining 4% are between 0.2 and 0.8 - but this means all the variance in 96% of the data isn't visible.
    """
    scale_idxs = np.where(preds > 0.2)[0]
    pred_min = preds[scale_idxs].min()
    pred_max = preds[scale_idxs].max()
    
    new_vals = (upper_bound-lower_bound) * (preds[scale_idxs] - pred_min) / (pred_max-pred_min) + lower_bound
    scaled_preds = np.copy(preds)
    scaled_preds[scale_idxs] = new_vals

    return scaled_preds
