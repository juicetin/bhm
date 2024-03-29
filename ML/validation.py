import numpy as np
from sklearn import cross_validation
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from ML.dir_mul.nicta.dirmultreg import dirmultreg_learn, dirmultreg_predict
from ML.gp.gp_gpy import GPyC
from ML.gp.poe import PoGPE
from ML.gp.gpoe import GPoGPE
from ML.gp.bcm import BCM
from ML.gp.rbcm import rBCM
from ML.gp.gp_multi_gpy import GPyMultiOutput
from ML import pseudo_multioutput
from ML import helpers
import pdb

def get_even_split(labels, threshold=20):
    bins = np.bincount(labels)
    per_label = int(0.9 * bins.min())
    if per_label < threshold:
        raise ValueError('each labl requires at least {} instances! Readjust/aggregate data to meet this requirement.'.format(threshold))
    train_idxs = []
    for i in range(bins.shape[0]):
        cur_label_idxs = np.where(labels == i)[0]
        np.random.shuffle(cur_label_idxs)
        train_idxs.append(cur_label_idxs[:per_label])
    train_idxs = np.concatenate(train_idxs)

    set(train_idxs)
    range(labels.shape[0])
    set(range(labels.shape[0]))
    test_idxs = np.array(list(set(range(labels.shape[0])) - set(train_idxs)))
    return train_idxs, test_idxs

def algo_module_to_str(algo):
    return str(algo()).split('(')[0]

def generate_cross_algo_print(*, algo_str, f1s, accs, label_cnt, auroc):
    # return generate_cross_algo_print(algo_str, f1s, accuracies, uniq_labels)
    f1_avg = np.around(np.average(f1s), decimals=5)
    acc_avg = np.around(np.average(accs), decimals=5)
    latex_row = '{} & {} & {}'.format(algo_str, f1_avg, acc_avg)
    if (auroc != None):
        latex_row += '& {}'.format(auroc)
    latex_row += '& {}\\\\n'.format(label_cnt + ' labels')
    return latex_row

def is_GP_model(module):
    valid_GP_models = np.array([GPyC, PoGPE, GPoGPE, BCM, rBCM])
    return (module == valid_GP_models).any()

def cross_validate_algo(features, labels, folds, algo, verbose=False):
    accuracies = []
    f1s = []
    # kf = cross_validation.KFold(n=len(features), n_folds=folds, shuffle=True, random_state=None)
    kf = cross_validation.StratifiedKFold(labels, n_folds=folds) # Prevent rounds with none of any given label - breaks AUC
    count = 1
    algo_str = algo_module_to_str(algo)
    f1_arrs = []
    uniq_c = np.unique(labels).shape[0]
    for train_index, test_index in kf:
    # for train_index, test_index in kf.split(features, labels):
        # print("Calculating part {} of {}".format(count, folds))
        count += 1
        # Break into training and test sets
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Execute classifier
        clf = algo()
        y_ = clf.fit(X_train, y_train).predict(X_test)

        #################### GP and GP ensemble handling ####################
        auroc=None
        if is_GP_model(algo):
            y_allpreds = y_[0]
            y_ = y_[0].argmax(axis=1)
            auroc = helpers.roc_auc_score_multi(y_test, y_allpreds)

        # Get scores
        this_accuracy = accuracy_score(y_test, y_)
        accuracies.append(this_accuracy)
        this_f1 = f1_score(y_test, y_, average=None)

        # missing values
        if this_f1.shape[0] != uniq_c:
            this_f1 = np.concatenate((this_f1, (uniq_c-this_f1.shape[0]) * [0] ))

        f1_arrs.append(this_f1)
        f1s.append(np.average(this_f1))

        if verbose == True:
            round_str = "This round's acc: {}, f1: {}, f1 avg: {}".format(this_accuracy, this_f1, np.average(this_f1))
            if auroc != None:
                round_str += ", auroc: {}, auroc avg: {}".format(auroc, np.average(auroc))
            print(round_str)

        del(clf)

    # return '{} & {} & {} & {} \\\\\n'.format(algo_str=algo_str, f1s=f1s, accs=accuracies, label_cnt=uniq_labels)
    label_cnt = str(np.unique(labels).shape[0])
    cross_algo_results = generate_cross_algo_print(algo_str=algo_str, f1s=f1s, accs=accuracies, label_cnt=label_cnt, auroc=auroc)
    return np.array(f1_arrs)

def cross_validate_dm_argmax(features, labels, algo, folds=10):
    accuracies = []
    f1s = []
    kf = cross_validation.KFold(n=len(features), n_folds=folds, shuffle=True, random_state=None)
    for train_idx, test_idx in kf:

        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx].argmax(axis=1)

        # Execute classifier
        y_ = algo.fit(X_train, y_train).predict(X_test).argmax(axis=1)
        # y_ = lr.fit(features[train_idx], labels[train_idx]).predict(features[test_idx])

        # Get scores
        this_accuracy = accuracy_score(y_test, y_)
        accuracies.append(this_accuracy)
        this_f1 = f1_score(y_test, y_, average=None)
        f1s.append(np.average(this_f1))

    print("Algo: {}, f1 avg: {}, acc avg: {}".format(str(algo), np.average(f1s), np.average(accuracies)))
    return str(algo), np.average(f1s), np.average(accuracies)

def cross_validate_algo_multioutput(features, labels, algo, folds=10):
    kf = cross_validation.KFold(n=len(features), n_folds=folds, shuffle=True, random_state=None)
    errs = []
    for train_idx, test_idx in kf:
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        y_ = pseudo_multioutput.predict(X_train, X_test, y_train, algo)
        errs.append(np.average(np.abs(y_test - y_)))

    return '{} & {} & {} & {} & {} & {}\\\\\n'.format(
            algo_module_to_str(algo), 
            np.round(np.average(errs), 4),
            labels.shape[1],
            np.round(np.average(y_.sum(axis=1)), 4),
            np.round(np.average(y_.sum(axis=1).min()), 4),
            np.round(np.average(y_.sum(axis=1).max()), 4)
            )

def cross_validate_dm(features, labels, folds=10):
    kf = cross_validation.KFold(n=len(features), n_folds=folds, shuffle=True, random_state=None)
    labels = labels/labels.sum(axis=1)[:,np.newaxis] # normalise

    errs = []

    for train_idx, test_idx in kf:
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        W = dirmultreg_learn(X_train, y_train)
        y_ = dirmultreg_predict(X_test, W)[0]

        all_err = y_ - y_test
        avg_err = np.average(np.abs(all_err))
        errs.append(avg_err)

        print(avg_err)

    # print("Average error was {}".format(np.average(errs)))
    orig_f = 'Original Features'
    quad_f = 'Quadratic projection'
    sq_f = 'Squared features with 1 bias'
    coords = 'using coordinates'
    no_coords = 'not using coordinates'
    coords_str_map = {
        9: '{}, {}'.format(orig_f, no_coords),
        11: '{}, {}'.format(orig_f, coords),
        19: '{}, {}'.format(sq_f, no_coords),
        23: '{}, {}'.format(sq_f, coords),
        55: '{}, {}'.format(quad_f, no_coords),
        78: '{}, {}'.format(quad_f, coords)
    }

    return '{} & {} \\\\\n'.format(coords_str_map[features.shape[1]], avg_err)

def cross_validate_gp_multi(features, labels, folds=10):
    kf = cross_validation.KFold(n=len(features), n_folds=folds, shuffle=True, random_state=None)
    errs = []

    for train_idx, test_idx in kf:
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        gp = GPyMultiOutput()
        gp.fit(X_train, y_train)

        y_ = gp.predict(X_test)[0].T

        all_err = y_ - y_test
        avg_err = np.average(np.abs(all_err))
        errs.append(avg_err)

        print(avg_err)

    print("Average error was {}".format(np.average(errs)))
