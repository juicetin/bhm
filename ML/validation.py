import numpy as np
from sklearn import cross_validation
import csv
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from ML.dir_mul.nicta.dirmultreg import dirmultreg_learn, dirmultreg_predict
import pdb

def cross_validate_algo(features, labels, folds, algo):
    accuracies = []
    f1s = []
    kf = cross_validation.KFold(n=len(features), n_folds=folds, shuffle=True, random_state=None)
    count = 1
    for train_index, test_index in kf:
        # print("Calculating part {} of {}".format(count, folds))
        count += 1
        # Break into training and test sets
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Execute classifier
        y_ = algo.fit(X_train, y_train).predict(X_test)
        # y_ = lr.fit(features[train_index], labels[train_index]).predict(features[test_index])

        # Get scores
        this_accuracy = accuracy_score(y_test, y_)
        accuracies.append(this_accuracy)
        this_f1 = f1_score(y_test, y_, average=None)
        f1s.append(np.average(this_f1))

        # print("f1: {}, acc: {}".format(np.average(this_f1), this_accuracy))
    print("Algo: {}, f1 avg: {}, acc avg: {}".format(str(algo), np.average(f1s), np.average(accuracies)))
    return str(algo), np.average(f1s), np.average(accuracies)

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

    print("Average error was {}".format(np.average(errs)))
