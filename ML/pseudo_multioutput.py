import numpy as np

def predict(train_features, test_features, labels, algo):
    preds = []
    for i in range(labels.shape[1]):
        model = algo()
        model.fit(train_features, labels[:,i].astype(int))
        preds.append(model.predict(test_features))

    return np.vstack(preds).T
