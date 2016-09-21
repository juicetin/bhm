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

def downsample(locations, features, labels):
    print("Downsampling data...")
    red_coords, red_features, red_mlabels = downsample_spatial_data(locations, features, labels, method='fixed-grid')
    ml_argsort = np.argsort(red_mlabels.sum(axis=1))
    return red_coords, red_features, red_mlabels, ml_argsort

