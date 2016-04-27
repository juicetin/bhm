import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

def accuracy(actual, predicted):
    correct = 0
    for (actual, predicted) in zip(actual, predicted):
        if actual == predicted:
            correct += 1
    return correct/len(actual)


# Naive k nearest neighbour classifier
def rf_once(features, label):
    folds=10
    accuracies = []
    kf = cross_validation.KFold(n=len(features), n_folds=folds, shuffle=True, random_state=None)
    data_size = len(features)
    label = np.array(list(map(str, label)))
    correct = 0
    for train_index, test_index in kf:
        # Break into training and test sets
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = label[train_index], label[test_index]

        # Execute rf classifier
        rf = RandomForestClassifier()
        y_ = rf.fit(X_train, y_train).predict(X_test)
        accuracies.append(accuracy(y_, y_test))
    avg_knn_accuracy = np.average(accuracies)
    return avg_knn_accuracy

# Run rf x times and get the average
def rf(features, label, n_times):
    print("Doing randomised 10-fold cross validation rf {} times".format(n_times))
    accuracies = []
    for i in range(0, n_times):
        accuracy_this_time = rf_once(features,label)
        accuracies.append(accuracy_this_time)
        print("On the {}-th go for randomise 10-fold rf, accuracy was {}%".format(i, accuracy_this_time))
    print("Average across all {} random 10-fold cross validated rfs was {}".format(n_times, np.average(accuracies)))

