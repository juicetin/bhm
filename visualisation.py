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

def show_map(locations, labels, x_bins, y_bins):
    x_bins.sort()
    y_bins.sort()

    x_min = min(x_bins)
    x_max = max(x_bins)
    y_min = min(y_bins)
    y_max = max(y_bins)

    print("Creating x, y index map to determine where to plot...")
    x_bin_coord_map = dict( zip( x_bins, range(len(x_bins)) ) )
    y_bin_coord_map = dict( zip( y_bins, range(len(y_bins)) ) )

    print("Building coordinate matrix with NaNs except where actual measurements exist...")
    X, Y = np.meshgrid(x_bins, y_bins)

    Z = np.zeros((X.shape[0], X.shape[1]))
    Z[:] = None
    x_locations = [x_bin_coord_map[x] for x, y in locations]
    y_locations = [y_bin_coord_map[y] for y, y in locations]
    Z[(y_locations, x_locations)] = labels

    print("Bulding image...")
    plt.imshow(Z, extent=[x_min, x_max, y_min, y_max])

    print("Setting colourbar (legend)...")
    plt.colorbar()

    print("Built! Showing image...")
    plt.show()
