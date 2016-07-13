import numpy as np
from matplotlib import pyplot as plt

def draw_map(X_train, X_test, y_train, y_test):
    pass

def plot(X, y, x, y_pred, sigma):
    # Plot function, prediction, and 95% confidence interval based on MSE
    confidence = 1.9600
    fig = plt.figure()
    # plt.plot(x, y, 'r:', label=u'$f(x) = x\, \sin(x)$')
    plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
    plt.plot(x, y_pred, 'b-', label=u'Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred - confidence * sigma,
                          (y_pred + confidence * sigma)[::-1]]),
            alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')
    plt.show()
