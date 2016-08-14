import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

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
    # plt.show()
    plt.save_fig('img.pdf')

def plot_classes(X, Y, Y_pred):
    if X.shape[1] == 3:
        x, y, z = X[:,0], X[:,1], X[:,2]

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, projection='3d')

        # ax3D_pred = fig.add_subplot(111, projection='3d')
        collection_p = ax1.scatter(x, y, z, c=Y_pred, cmap = cm.Spectral)

        # ax3D_r = fig.add_subplot(121, projection='3d', sharex=ax3D_pred)
        collection_r = ax2.scatter(x, y, z, c=Y, cmap=cm.Spectral)

        plt.colorbar(collection_p)
        plt.show()

    elif X.shape[1] == 2:
        x, y, = X[:,0], X[:,1]
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

        # Predictions
        # Preds Plots
        pos_idx = np.nonzero(Y_pred==1)[0]
        neg_idx = np.array(list(set(list(np.arange(len(Y)))) - set(list(pos_idx))))
        x_pos = x[pos_idx]
        y_pos = y[pos_idx]
        x_neg = x[neg_idx]
        y_neg = y[neg_idx]

        Y_pred_pos = Y_pred[pos_idx]
        Y_pred_neg = Y_pred[neg_idx]
        # ax1.scatter(x, y, c=Y_pred)

        ax1.plot(x_pos, y_pos, 'o', color='y')
        ax1.plot(x_neg, y_neg, 'x', color='r')
        ax1.set_title('Predictions')

        # Actuals Plots
        pos_idx = np.nonzero(Y==1)[0]
        neg_idx = np.array(list(set(list(np.arange(len(Y)))) - set(list(pos_idx))))
        x_pos = x[pos_idx]
        y_pos = y[pos_idx]
        x_neg = x[neg_idx]
        y_neg = y[neg_idx]

        Y_pos = Y[pos_idx]
        Y_neg = Y[neg_idx]
        # ax2.scatter(x, y, c=Y)
        ax2.plot(x_pos, y_pos, 'o', color='y')
        ax2.plot(x_neg, y_neg, 'x', color='r')
        ax2.set_title('Observations')

        plt.show()

def show_map(locations, labels, x_bins, y_bins, display=True):
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
    plt.imshow(Z, extent=[x_min, x_max, y_min, y_max], origin='lower')

    print("Setting colourbar (legend)...")
    plt.colorbar()

    print("Image generated!")
    if display == True:
        plt.show()
    else:
        plt.savefig('map.pdf')
