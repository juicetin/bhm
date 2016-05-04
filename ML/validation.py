import numpy as np
from sklearn import cross_validation
import csv
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def cross_validate_algo(features, labels, folds, algo):
    folds = 10
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
        this_f1 = f1_score(y_test, y_, average=None)
        accuracies.append(this_accuracy)
        f1s.append(np.average(this_f1))

        # print("f1: {}, acc: {}".format(np.average(this_f1), this_accuracy))
    print("Algo: {}, f1 avg: {}, acc avg: {}".format(str(algo), np.average(f1s), np.average(accuracies)))
    return np.average(f1s), np.average(accuracies)

