import numpy as np
from datetime import datetime
import matplotlib as mpl

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
# from utils.benchmarks import dm_test_data
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib import colors
import math
import pdb

from utils.downsample import fixed_grid_blocksize

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

def show_map(locations, labels, x_bins=None, y_bins=None, display=True, filename='map', vmin=None, vmax=None):
    """
    Given the x, y coord locations and corresponding labels, plot this on imshow (null points
    will be shown as blank in the background).
    """

    if (x_bins == None and y_bins == None):
        # TODO built xbins and ybins
        # ax_coords = np.arange(-7, 7.2, 0.2)
        # x, y = np.meshgrid(ax_coords, ax_coords)

        # x_bins = np.concatenate((np.unique(locations[:,0]), ax_coords))
        # y_bins = np.concatenate((np.unique(locations[:,1]), ax_coords))

        x_bins = np.unique(locations[:,0])
        y_bins = np.unique(locations[:,1])

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

    print("Setting colourbar (legend)...")
    # cmap = cm.jet
    # cmaplist = [cmap(i) for i in range(cmap.N)]
    # cmap = cmap.from_list('custom cmap', cmaplist, cmap.N)

    print("Bulding image...")
    # plt.imshow(Z, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.imshow(Z, extent=[x_min, x_max, y_min, y_max], origin='lower', vmin=vmin, vmax=vmax)

    # Slightly hacky - unfortunately neeeded for 0-count argmaxs of 24 labels
    # if np.unique(labels).shape[0] < 5:
    #     uniq_labels = np.arange(5)
    # else:
    #     uniq_labels = np.arange(25)

    # bounds = np.linspace(uniq_labels.min()+1, uniq_labels.max(), uniq_labels.shape[0])
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # plt.colorbar(cmap=cmap, norm=norm, spacing='Proportional', boundaries=bounds, format='%li')
    plt.colorbar(spacing='Proportional', format='%li')

    print("Image generated!")
    if display == True:
        plt.show()
    else:
        plt.savefig(filename + '.pdf')

    plt.cla()
    plt.clf()

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

def histogram(freqs, title=None, filename='freqs.pdf', offset=0):
    """
    Plots a histogram 
    """
    bins = np.arange(offset, freqs.shape[0]+offset)
    plt.hist(bins, bins=bins - .5, weights=freqs, lw=0, color=['blue'])
    plt.xticks(bins[:-1])

    if title != None:
        plt.title(title)
    # for i, txt in enumerate(freqs):
    #     plt.annotate(str(txt), (i, 0), xytext=(i,-300), va='top', ha='center')
    plt.savefig(filename)

def clear_plt():
    """
    Clears pyplot caches/etc.
    """
    plt.cla()
    plt.clf()

def plot_coords(locations, filename='tmp.pdf', display=True):
    """
    Plots a given set of x,y coordinates.
    Locations are given as a list of (x,y) tuples
    """
    x = locations[:,0]
    y = locations[:,1]
    plt.scatter(x, y)
    if display == False:
        plt.savefig(filename)
    else:
        plt.show()

def plot_training_with_grid(locations, filename='training_map.pdf', display=True, reduction_factor=2):
    """
    Plots the training datapoints with fixed downsampling grid overlaid on top
    """
    _, _, _, _, _, _, reduced_x_coords, reduced_y_coords = fixed_grid_blocksize(locations, reduction_factor)
    fig = plt.figure()
    ax = fig.gca()
    # ax.set_xticks(reduced_x_coords)
    # ax.set_yticks(reduced_y_coords)
    # ax.set_yticks([])
    # ax.set_xticks([375000])
    plt.scatter(locations[:,0], locations[:,1])
    plt.grid(linestyle='dashed')
    plt.savefig(filename)

    pdb.set_trace()


def plot_toy_data(locations, colours, title='Illustrative example plots', filename='tmp.pdf', display=True):
    """
    Plot the toy DM vs GP data to show clusters
    """
    x = locations[:,0]
    y = locations[:,1]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(x, y, c=colours, lw=0)
    ax.set_title(title)
    ax.annotate('cluster A', xy=(-4, 2.2), xytext=(-3, 0),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
    ax.annotate('cluster B', xy=(5.5, 2), xytext=(3, -1),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

    ax.annotate('cluster C', xy=(-3, -7), xytext=(-1, -9),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

    # Show image
    if display == False:
        plt.savefig(filename)
    else:
        plt.show()
    clear_plt()

def plot_multilabel_distribution(labels, title='Multi-label distribution', filename='multilabel_distr.pdf', display=True):
    """
    Plot a histogram of the distribution of multi-labels (a single set)
    """
    x = np.arange(labels.shape[0]) # x-axis
    y1 = labels[:,0] # first label
    y2 = labels[:,1] # second label

    # fig = plt.figure() 
    # ax = fig.add_subplot(111)

    # ax.scatter(x, y1, c='b')
    # ax.scatter(x, y2, c='r')

    p1 = plt.bar(x, y1, color='r',  lw=0)
    p2 = plt.bar(x, y2, bottom=y1, color='b', lw=0)

    plt.title(title)

    # Show image
    if display == False:
        plt.savefig(filename)
    else:
        plt.show()

    clear_plt()

def dm_pred_vs_actual(preds, actuals, title='DM predictions vs actuals', filename='dm_pred_plot', display=False):
    """
    Plots all the DM predictions vs actual for the distribution of labels at each point
    """
    x = np.arange(1, preds.shape[0]+1)
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'b', 'w']
    # cmap = cm.jet
    # colours = [cmap(i) for i in range(cmap.N)]

    for i in range(preds.shape[1]):
        cur_preds = preds[:,i]
        cur_actuals = actuals[:,i]
        pred_scat = plt.scatter(x, cur_preds, marker='x', c=colours[i])
        actual_scat = plt.scatter(x, cur_actuals, marker='o', c=colours[i])
        plt.legend([pred_scat, actual_scat], ['Predictions', 'Actuals'])
        plt.title(title + str(i))

        if display == False:
            plt.savefig(filename+str(i)+'.pdf')
        clear_plt()

def gp_pred_vs_actual(y_distr, y_pred, sigma, display=False, filename='toy_gp_pred_plot_'):
    """
    Plots the predictions of a GP for a particular class (binary/OvA) with the variance highlighted
    """
    x = np.arange(1, y_distr.shape[0]+1)
    for i in range(y_pred.shape[0]):
        # Create y where positive is the 'current' label
        y = [1 if max(row) == row[i] else 0 for row in y_distr]

        # Test actuals
        test_actuals = plt.plot(x, y, 'rx', markersize=2, label=u'Test', mew=2.0)
    
        # Predictions and variance
        predictions = plt.plot(x, y_pred[i], 'g-', label=u'Prediction', mew=2.0)
        predictions = plt.plot(x, y_pred[i], 'g-', label=u'Prediction', mew=2.0)
        variance = plt.fill(np.concatenate([x, x[::-1]]),
                np.concatenate([y_pred[i] - sigma[i],
                    (y_pred[i] + sigma[i])[::-1]]),
                alpha=0.2, fc='b', ec='None', label='95% confidence interval')
    
        # Axes labels
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.legend([test_actuals[0], predictions[0], variance[0]], ['Test', 'Raw Predictions', 'Variance'])
    
        # plt.legend(loc='upper left')
        if display == False:
            plt.savefig(filename+str(i)+'.pdf')
        else:
            plt.show()
        clear_plt()

def standalone_toyplot_hist_legend(filename='toyplot_hist_distr_legend.pdf'):
    """
    Generates a standalone legend describing the histogram of the toy plots 
    """
    fig = plt.figure()
    figlegend = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    bar1 = ax.bar(range(10), np.random.randn(10), color='r')
    bar2 = ax.bar(range(10), np.random.randn(10), color='b')
    x_descript = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    y_descript = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    figlegend.legend([x_descript, y_descript, bar1, bar2], ('x-axis - normalised ratio of label', 'y-axis - data point number', 'Label 0', 'Label 1'), 'center')

    figlegend.savefig(filename, bbox_inches='tight', pad_inches = 0)

def standalone_DM_colorbar_legend(filename='dm_plot_colorbar.pdf'):
    """
    Generates a standalone colorbar for the DM heatmaps
    """
    fig = plt.figure()
    figlegend = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    bar1 = ax.bar(range(10), np.random.randn(10), color='r')
    bar2 = ax.bar(range(10), np.random.randn(10), color='b')
    x_descript = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    y_descript = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    figlegend.legend([x_descript, y_descript, bar1, bar2], ('x-axis - normalised ratio of label', 'y-axis - data point number', 'Label 0', 'Label 1'), 'center')

    figlegend.savefig(filename, bbox_inches='tight', pad_inches = 0)

def plot_dm_chains(chains, filename='dm_mcmc_weights'):
    label_count = chains.shape[1]
    x = np.arange(1, chains.shape[0]+1)
    rows = math.ceil(math.sqrt(chains.shape[1]))
    # axs = generate_subplots(rows=rows, columns=rows, actual_count=label_count, title_list=None)
    for idx in range(label_count):
        plt.plot(x, chains[:,idx])
        plt.savefig(filename+'_'+str(idx)+'.pdf')
        clear_plt()

def plot_dm_hists(chains, filename='dm_mcmc_weight_hist'):
    pass
