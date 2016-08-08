import numpy as np

# Sklearn
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedShuffleSplit

def partition_indexes(length, blocks):
    block_size = int(length/blocks)
    idxs = []
    for i in range(blocks):
        idxs.append((i*block_size, (i+1)*block_size))
    idxs[-1] = (idxs[-1][0], length)
    return idxs

def sigmoid(x):
    return 1/(1+np.exp(-x))

def score(y_, y):
    return f1_score(y, y_, average='weighted')

def roc_auc_score_multi(y_actuals, y_preds):

    # Calculate AUROC for each each binary class case
    aurocs = np.zeros(y_preds.shape[0])
    for cur_class, cur_ova_pred in enumerate(y_preds):
        # Compare class *i* with the rest
        cur_y_actual = np.copy(y_actuals)
        cur_y_actual[np.where(cur_y_actual != cur_class)] = -1
        cur_y_actual[np.where(cur_y_actual != -1)] = 1
        cur_auroc = roc_auc_score(cur_y_actual, cur_ova_pred)
        aurocs[cur_class] = cur_auroc

    print("AUROC score for each class: {}".format(aurocs))
    # TODO weight AUROCS in future?
    return np.average(aurocs)

def stratified_data_idxs(labels, blocks):
    sss = StratifiedShuffleSplit(labels, n_iter=1, test_size=1-1/blocks)
    cur_blocks = blocks
    test_size = 1/cur_blocks
    strat_idx = np.array([])
    big_idxs = np.arange(labels.shape[0])
    while cur_blocks > 1:
        sss = StratifiedShuffleSplit(labels[big_idxs], n_iter=1, test_size=test_size)
        both_idxs = [(big_idx, sml_idx) for big_idx, sml_idx in sss][0]
        print("Training, test indexes: {}".format(both_idxs))
        big_idxs = both_idxs[0]
        sml_idxs = both_idxs[1]
        strat_idx = np.concatenate((strat_idx, sml_idxs), axis=0)
        cur_blocks -= 1

    # Account for last block
    strat_idx = np.concatenate((strat_idx, big_idxs))

    return strat_idx
