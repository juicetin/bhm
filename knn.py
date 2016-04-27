import numpy as np
from sklearn import neighbors
from sklearn import cross_validation

def accuracy(actual, predicted):
    correct = 0
    for (actual, predicted) in zip(actual, predicted):
        if actual == predicted:
            correct += 1
    return correct/len(actual)


# Naive k nearest neighbour classifier
def kNN_once(features, label):
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

        # Execute kNN classifier
        knn = neighbors.KNeighborsClassifier(n_neighbors=5)
        y_ = knn.fit(X_train, y_train).predict(X_test)
        accuracies.append(accuracy(y_, y_test))
    avg_knn_accuracy = np.average(accuracies)
    return avg_knn_accuracy

# Run kNN x times and get the average
def kNN(features, label, n_times):
    print("Doing randomised 10-fold cross validation kNN {} times".format(n_times))
    accuracies = []
    for i in range(0, n_times):
        accuracy_this_time = kNN_once(features,label)
        accuracies.append(accuracy_this_time)
        print("On the {}-th go for randomise 10-fold kNN, accuracy was {}%".format(i, accuracy_this_time))
    print("Average across all {} random 10-fold cross validated kNNs was {}".format(n_times, np.average(accuracies)))
