import GPy
import numpy as np
import pdb
from ML.helpers import roc_auc_score_multi
from ML.helpers import binarised_labels_copy
from ML.helpers import regression_score

def gpy_bin_predict(features, labels):
    m = GPy.models.GPClassification(features[train_idx], labels[train_idx])
    probs = m.predict(features_sn[test_idx])[0].T[0,:]

def gpy_bench(features, labels, train_idx):
    test_idx = np.array(list(set(np.arange(16502)) - set(train_idx)))
    if (len(labels.shape) == 1):
        labels = labels.reshape(labels.shape[0], 1)

    pred_probs = []
    uniq_labels = np.unique(labels)
    print("building GPy model for class...", end="", flush=True)
    kernel = GPy.kern.RBF(input_dim=features.shape[1])
    for c in uniq_labels:
        print(c, end=" ", flush=True)
        cur_bin_labels = binarised_labels_copy(labels, c)
        m = GPy.models.GPClassification(features[train_idx], cur_bin_labels[train_idx], kernel=kernel)
        probs = m.predict(features[test_idx])[0].T[0,:]
        pred_probs.append(probs)
    print()

    pred_probs = np.array(pred_probs).reshape(uniq_labels.shape[0], test_idx.shape[0])
    return pred_probs

def test():
    #This functions generate data corresponding to two outputs
    f_output1 = lambda x: 4. * np.cos(x/5.) - .4*x - 35. + np.random.rand(x.size)[:,None] * 2.
    f_output2 = lambda x: 6. * np.cos(x/5.) + .2*x + 35. + np.random.rand(x.size)[:,None] * 8.

    #{X,Y} training set for each output
    X1 = np.random.rand(100)[:,None]; X1=X1*75
    X2 = np.random.rand(100)[:,None]; X2=X2*70 + 30
    Y1 = f_output1(X1)
    Y2 = f_output2(X2)
    #{X,Y} test set for each output
    Xt1 = np.random.rand(100)[:,None]*100
    Xt2 = np.random.rand(100)[:,None]*100
    Yt1 = f_output1(Xt1)
    Yt2 = f_output2(Xt2)

    xlim = (0,100); ylim = (0,50)

    import matplotlib.pyplot as plt
    K = GPy.kern.Matern32(1)

    m1 = GPy.models.GPRegression(X1,Y1,kernel=K.copy())
    m1.optimize()
    m2 = GPy.models.GPRegression(X2,Y2,kernel=K.copy())
    m2.optimize()

    y,v = m1.predict(Xt1)
    print(regression_score(Yt1, y))

    print(m1)

    fig = plt.figure(figsize=(12,8))
    #Output 1
    ax1 = fig.add_subplot(211)
    m1.plot(plot_limits=xlim,ax=ax1)
    ax1.plot(Xt1[:,:1],Yt1,'rx',mew=1.5)
    ax1.set_title('Output 1')
    #Output 2
    ax2 = fig.add_subplot(212)
    m2.plot(plot_limits=xlim,ax=ax2)
    ax2.plot(Xt2[:,:1],Yt2,'rx',mew=1.5)
    ax2.set_title('Output 2')


    pdb.set_trace()
    plt.show()

    return m1
