import GPy
import numpy as np
from ML.helpers import roc_auc_score_multi
from ML.helpers import binarised_labels_copy

def gpy_bin_predict(features, labels):
    m = GPy.models.GPClassification(features[train_idx], labels[train_idx])
    probs = m.predict(features_sn[test_idx])[0].T[0,:]

def gpy_bench(features, labels, train_idx):
    test_idx = np.array(list(set(np.arange(16502)) - set(train_idx)))
    if (len(labels.shape) == 1):
        labels = labels.reshape(labels.shape[0], 1)

    pred_probs = []
    uniq_labels = np.unique(labels)
    print("building GPy model for class...", end="", flush=True)
    kernel = GPy.kern.RBF(input_dim=features.shape[1])
    for c in uniq_labels:
        print(c, end=" ", flush=True)
        cur_bin_labels = binarised_labels_copy(labels, c)
        m = GPy.models.GPClassification(features[train_idx], cur_bin_labels[train_idx], kernel=kernel)
        probs = m.predict(features[test_idx])[0].T[0,:]
        pred_probs.append(probs)
    print()

    pred_probs = np.array(pred_probs).reshape(uniq_labels.shape[0], test_idx.shape[0])
    return pred_probs
