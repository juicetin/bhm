import numpy as np
from datetime import datetime
import matplotlib as mpl

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from misc.benchmarks import dm_test_data
import matplotlib.cm as cm
import pdb

def draw_map(X_train, X_test, y_train, y_test):
    pass

def plot(X, Y, x, y, y_pred, sigma):

    # Plot function, prediction, and 95% confidence interval based on MSE
    confidence = 1.9600
    fig = plt.figure(figsize=(12,8))
    print('got here')

    # plt.plot(x, y, 'r:', label=u'$f(x) = x\, \sin(x)$')

    # Training
    plt.plot(X, Y, 'bo', markersize=5, label=u'Observations')

    # Test actuals
    plt.plot(x, y, 'rx', markersize=5, label=u'Test', mew=2.0)

    # Predictions and variance
    plt.plot(x, y_pred, 'g-', label=u'Prediction', mew=2.0)
    plt.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred - confidence * sigma,
                          (y_pred + confidence * sigma)[::-1]]),
            alpha=0.2, fc='b', ec='None', label='95% confidence interval')

    # Axes labels
    plt.xlabel('$x$')
    plt.ylabel('$y$')

    # Handle logic of bounding graph on min/max of x/y coordinates
    y_mins = 1.1*np.array([np.min(Y), np.min(y), np.min(y_pred)])
    y_maxs = 1.1*np.array([np.max(Y), np.max(y), np.max(y_pred)])
    x_mins = 1.05*np.array([np.min(X), np.min(x)])
    x_maxs = 1.05*np.array([np.max(X), np.max(x)])
    plt.ylim(np.min(y_mins), np.max(y_maxs))
    plt.xlim(np.min(x_mins), np.max(x_maxs))

    # plt.legend(loc='upper left')
    # plt.show()

def show_all():
    # In a display backend
    backend = mpl.get_backend()
    print("Using backend {}".format(backend))
    if backend != 'agg':
        plt.show()

    # No display backend
    else:
        date = datetime.now()
        plt.savefig('images/{}.pdf'.format(date))

def add_confidence_plot(ax, x, y, sigma):
    confidence = 1.9600
    ax.plot(x, y, 'b-', label=u'Prediction')
    ax.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y - confidence * sigma,
                           (y + confidence * sigma)[::-1]]),
            alpha=.5, fc='b', ec='None', label='95% confidence interval')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

def add_scatter_plot(ax, x, y):
    colors = np.ones_like(x)
    ax.scatter(x, y, c=colors, marker='o')

def generate_subplots(rows=1, columns=1, actual_count=1, title_list=None):
    fig = plt.figure()
    axs = [fig.add_subplot(rows, columns, i) for i in range(1, actual_count+1)]
    if title_list != None:
        for title, ax in zip(title_list, axs):
            ax.set_title(title)
            # ax.set_xlim(-2, 3)
            # ax.set_ylim(-2, 2)

    return axs

def plot_confidence(x, y_pred, sigma, title=None):
    # Plot function, prediction, and 95% confidence interval based on MSE
    confidence = 1.9600
    fig = plt.figure()
    # plt.plot(x, y, 'r:', label=u'$f(x) = x\, \sin(x)$')
    plt.plot(x, y_pred, 'b-', label=u'Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred - confidence * sigma,
                          (y_pred + confidence * sigma)[::-1]]),
            alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$y$')

    plt.ylim(-200, 200)

    plt.legend(loc='upper left')
    if title != None:
        plt.title(title)
    plt.show()

def plot_test_graph():
    f_output1 = lambda x: 4. * np.cos(x/5.) - .4*x - 35. + np.random.rand(x.size)[:,None] * 2
    X1 = np.random.rand(100)[:,None]; X1=X1*75
    Y1 = f_output1(X1)
    fig = plt.figure(1)
    ax1 = fig.add_subplot(211)
    ax1.scatter(X1, Y1)
    # ax1.set_xlim(2, 3)
    # ax1.set_ylim(-5, 5)
    plt.show()

def plot_continuous(X, Y):
    if X.shape[1] == 2:
        x, y, z = X[:,0], X[:,1], Y

        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')

        collection = ax1.scatter(x, y, z)
        plt.show()

    elif X.shape[1] == 1:
        x, y = X, Y

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        collection = ax1.scatter(x, y)
        plt.show()

def plot_classes(X, Y, Y_pred):
    if X.shape[1] == 3:
        x, y, z = X[:,0], X[:,1], X[:,2]

        # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, projection='3d')
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax2 = fig.add_subplot(121, projection='3d')

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

def multi_label_histogram(multi_labels):
    """
    Plots a histogram of how many classifications are contained within each label set
    """
    non_zero_labels = np.array([np.sum(labels != 0) for labels in multi_labels])
    bins = np.bincount(non_zero_labels)
    # plt.hist(bins[1:], bins.shape[0]-1)
    plt.hist(non_zero_labels, bins=range(1,bins.shape[0]), bottom=1)
    for i, txt in enumerate(bins):
        plt.annotate(str(txt), (i, 0), xytext=(i,-300), va='top', ha='center')
    pdb.set_trace()
    plt.savefig('label_occurrences_full24classes.pdf')

def plot_coords(locations):
    """
    Plots a given set of x,y coordinates.
    Locations are given as a list of (x,y) tuples
    """
    x = locations[:,0]
    y = locations[:,1]
    plt.scatter(x, y)
    plt.show()

# def dm_test_data():
#     X,C = dm_test_data()
#     pdb.set_trace()
#     plt.scatter(X[:,0], X[:,1])
#     plt.savefig('tmp.pdf')
