import numpy as np
from sklearn import cross_validation
import csv
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from ML.dir_mul.nicta.dirmultreg import dirmultreg_learn, dirmultreg_predict
from ML.gp.gp_gpy import GPyC
from ML.gp.gp_multi_gpy import GPyMultiOutput
import pdb

def cross_validate_algo(features, labels, folds, algo, verbose=False):
    accuracies = []
    f1s = []
    kf = cross_validation.KFold(n=len(features), n_folds=folds, shuffle=True, random_state=None)
    count = 1
    algo_str = str(algo()).split('(')[0]
    for train_index, test_index in kf:
        # print("Calculating part {} of {}".format(count, folds))
        count += 1
        # Break into training and test sets
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Execute classifier
        clf = algo()
        y_ = clf.fit(X_train, y_train).predict(X_test)
        # y_ = lr.fit(features[train_index], labels[train_index]).predict(features[test_index])

        # Account for when dealing with GP (TODO any probablistic-type outputs)
        if type(clf) == GPyC:
            y_ = y_[0].argmax(axis=1)

        # Get scores
        this_accuracy = accuracy_score(y_test, y_)
        accuracies.append(this_accuracy)
        this_f1 = f1_score(y_test, y_, average=None)
        f1s.append(np.average(this_f1))

        if verbose == True:
            print("This round's f1: {}, acc: {}".format(this_accuracy, this_f1))

        del(clf)

        # print("f1: {}, acc: {}".format(np.average(this_f1), this_accuracy))
    # print("Algo: {}, f1 avg: {}, acc avg: {}".format(str(algo), np.average(f1s), np.average(accuracies)))
    f1_avg = np.around(np.average(f1s), decimals=5)
    acc_avg = np.around(np.average(accuracies), decimals=5)
    algo_name = str(np.unique(labels).shape[0])
    return '{} & {} & {} & {} \\\\\n'.format(algo_str, f1_avg, acc_avg, algo_name + ' labels')

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

def cross_validate_dm(features, labels, folds=10):
    kf = cross_validation.KFold(n=len(features), n_folds=folds, shuffle=True, random_state=None)
    labels = labels/labels.sum(axis=1)[:,np.newaxis] # normalise

    errs = []

    for train_idx, test_idx in kf:
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        W = dirmultreg_learn(X_train, y_train)
        y_ = dirmultreg_predict(X_test, W)[0]

        all_err = y_ - labels[test_idx]
        avg_err = np.average(np.abs(all_err))
        errs.append(avg_err)

        print(avg_err)

    # print("Average error was {}".format(np.average(errs)))
    if features.shape[1] == 9 or features.shape[1] == 19:
        coords_str = 'using coordinates'
    else:
        coords_str = 'not using coordinates'

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
