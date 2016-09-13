import numpy as np
import copy
import sys
import math
import pdb
from datetime import datetime
import GPy

from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import LeaveOneOut
from sklearn.cross_validation import StratifiedKFold
from sklearn import datasets
from sklearn.metrics import roc_auc_score
from sklearn import neighbors

# Import our ML algorithms
from ML.validation import cross_validate_algo
from ML.knn import kNN
from ML.random_forests import rf
from ML.logistic_regression import lr

from ML import helpers
from ML.gp.gp import GaussianProcess
from ML.gp.poe import PoGPE
from ML.gp.gpoe import GPoGPE
from ML.gp.bcm import BCM
from ML.gp.rbcm import rBCM
from ML.gp.gp_mt import GPMT
from ML.dir_mul.dirichlet_multinomial import DirichletMultinomialRegression

import utils.visualisation as vis
import utils.load_data as data
import utils.gpy_benchmark

import pdb
import code

def dir_mul_bench(gen_maps=False):
    from matplotlib import pyplot as plt
    X, C = dm_test_data() 
    C_max = np.argmax(C, axis=1)

    from ML.dir_mul.nicta.dirmultreg import dirmultreg_learn, dirmultreg_predict
    W = dirmultreg_learn(X, C, activation='exp')


    dm = DirichletMultinomialRegression()
    dm.fit(X,C)

    preds_me_orig = dm.predict(X)
    preds_me_orig_max = np.argmax(preds_me_orig, axis=1)
    preds_nicta_orig = dirmultreg_predict(X, W, activation='soft')[0]
    preds_nicta_orig_max = np.argmax(preds_nicta_orig, axis=1)

    print("Origs: {}\nMpreds:{}\nOpreds:{}\n".format(C_max, preds_me_orig_max, preds_nicta_orig_max))

    # (-7, 7), (7, 7), (-7, -7), (7, -7)
    ax_coords = np.arange(-7, 7.2, 0.2)
    x, y = np.meshgrid(ax_coords, ax_coords)
    coords = np.array([[xc, yc] for xc, yc in zip(np.concatenate(x), np.concatenate(y))])

    preds_nicta, _ = dirmultreg_predict(coords, W, activation='soft')
    preds_nicta_max = np.argmax(preds_nicta, axis=1)

    preds_me = dm.predict(coords)
    preds_me_max = np.argmax(preds_me, axis=1)

    if gen_maps == True:
        path='images/toydata/'
        vmin = 0
        vmax = 2
        vis.show_map(X, C_max, display=False, filename=path+'toydata_orig', vmin=vmin, vmax=vmax)
        vis.show_map(X, preds_me_orig_max, display=False, filename=path+'toydata_preds_orig_me', vmin=vmin, vmax=vmax)
        vis.show_map(X, preds_nicta_orig_max, display=False, filename=path+'toydata_preds_orig_oth', vmin=vmin, vmax=vmax)
        vis.show_map(coords, preds_me_max, display=False, filename=path+'toydata_full_preds_me', vmin=vmin, vmax=vmax)
        vis.show_map(coords, preds_nicta_max, display=False, filename=path+'toydata_full_preds_oth', vmin=vmin, vmax=vmax)

    print("Entering interactive mode in benchmarks.dir_mul_bench")
    code.interact(local=locals())

def test():
    f_output1 = lambda x: 4. * np.cos(x/5.) - .4*x - 35. + np.random.rand(x.size)[:,None] * 2.
    f_output2 = lambda x: 6. * np.cos(x/5.) + .2*x + 35. + np.random.rand(x.size)[:,None] * 8.

    #{X,Y} training set for each output
    X1 = np.random.rand(100)[:,None]; X1=X1*75; X1.sort(axis=0)
    X2 = np.random.rand(100)[:,None]; X2=X2*70 + 30; X2.sort(axis=0)
    Y1 = f_output1(X1)
    Y2 = f_output2(X2)
    #{X,Y} test set for each output
    Xt1 = np.random.rand(100)[:,None]*100; Xt1.sort(axis=0)
    Xt2 = np.random.rand(100)[:,None]*100; Xt2.sort(axis=0)
    Yt1 = f_output1(Xt1)
    Yt2 = f_output2(Xt2)

    # pdb.set_trace()

    # gp = GaussianProcess()
    gp = GPMT()
    # gp = BCM(50)
    gp.fit(X1, Y1)
    y, v = gp.predict(Xt1)
    score = helpers.regression_score(Yt1, y)
    print(score)

    # m = GPy.models.GPCoregionalizedRegression(X1, Y1)
    # # m.optimize()
    # pdb.set_trace()
    # # y, v = m.predict(Xt1)
    # score = helpers.regression_score(Yt1, y)
    # print(score)

    vis.plot(X1, Y1, Xt1, Yt1, y, v)
    vis.show_all()

    # gp2 = GaussianProcess()
    # gp2.fit(X2, Y2)
    # y2, v2 = gp2.predict(Xt2)
    # score = helpers.regression_score(Yt2, y2)
    # print(score)
    # vis.plot(X2, Y2, Xt2, Yt2, y2, v2)

    # if show_plots == True:
    #     vis.show_all()

    return gp

    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(12,8))
    # ax1 = fig.add_subplot(111)
    # ax1.scatter(X1, Y1)
    # ax1.plot(Xt1[:,:1],Yt1,'rx',mew=1.5)
    # ax1.set_title('Output 1')
    # plt.show()


def test_2D_data_for_model(gp_model, ax, X_train, y_train, X_test):
    gp_model.fit(X_train, y_train)
    preds, variances = gp_model.predict(X_test)

    vis.add_confidence_plot(ax, X_test, preds, variances)
    vis.add_scatter_plot(ax, X_train, y_train)

def test_basic_2D_data():
    # t = np.arange(0.00, 1.5, 0.01)
    # noise = np.random.normal(0, 0.01, t.shape[0])
    # y = np.sin(0.2*np.pi*t) + noise
    # y = np.sin(6.5*np.pi*t) + np.cos(8.5*np.pi*t) + t - t**2
    # y = np.sin(np.pi*t) + np.cos(np.pi*t)

    f_output1 = lambda x: 4. * np.cos(x/5.) - .4*x - 35. + np.random.rand(x.size)[:,None] * 2
    t = np.random.rand(100)[:,None]; t=t*75
    y = f_output1(t)
    import matplotlib.pyplot as plt
    fig = plt.figure(1)
    ax1 = fig.add_subplot(211)
    ax1.scatter(t, y)
    plt.show()

    t = t.reshape(len(t), 1)
    y = y.reshape(len(y), 1)

    title_list=['gp', 'poe', 'gpoe', 'bcm', 'rbcm', 'GPy']
    axs = vis.generate_subplots(rows=2, columns=3, actual_count=6, title_list=title_list)

    # X_train, X_test = t[0::5], np.concatenate((empty_points, t[0::5]))

    train_idx = np.arange(0, t.shape[0], 15)
    test_idx = np.array(list(set(np.arange(t.shape[0])) - set(train_idx)))
    print(train_idx.shape)


    # empty_points = np.arange(-1.5, 0, 0.01).reshape(int(1.5/0.01), 1)
    # X_train, X_test = t[train_idx], np.concatenate((empty_points, t[test_idx]))
    X_train, X_test = t[train_idx], t[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # vis.plot_continuous(X_test, y_test)

    expert_size = train_idx.shape[0]/3

    # Own implementations
    test_2D_data_for_model(GaussianProcess(), axs[0], X_train, y_train, X_test)
    test_2D_data_for_model(PoGPE(expert_size), axs[1], X_train, y_train, X_test)
    test_2D_data_for_model(GPoGPE(expert_size), axs[2], X_train, y_train, X_test)
    test_2D_data_for_model(BCM(expert_size), axs[3], X_train, y_train, X_test)
    test_2D_data_for_model(rBCM(expert_size), axs[4], X_train, y_train, X_test)

    # GPy
    K = GPy.kern.Matern32(1)
    m = GPy.models.GPRegression(X_train, y_train, kernel=K.copy())
    preds, var = m.predict(X_test)
    vis.add_confidence_plot(axs[5], X_test, preds, var)
    vis.add_scatter_plot(axs[5], X_train, y_train)


    vis.show_all()

def classification_bathy_testing(features, labels):
    ####################################################################################
    # GP Classification on Bathymetry Data
    ####################################################################################

    gp = GaussianProcess()

    # cv = LeaveOneOut(len(labels))
    cv = StratifiedKFold(labels, n_folds=10)
    # cv = StratifiedShuffleSplit(labels, 1, test_size=0.1)
    scores = []
    for train_index, test_index in cv:
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        gp.fit(X_train, y_train)
        y_pred = gp.predict(X_test, keep_probs=True)
        # return y_pred

        score = helpers.roc_auc_score_multi(y_test, y_pred)
        # score = roc_auc_score(y_test, y_pred)
        scores.append(score)
        print("Average AUROC score for this round is: {}".format(score))

    avg_score = np.average(scores)
    print("Average AUROC score: {}".format(avg_score))
    return avg_score

def regression_dummy_testing():
    ####################################################################################
    # Dummy testing - regression
    ####################################################################################

    X, y = datasets.make_regression(n_samples=1000, n_features=2)
    print(type(y[0]))

    iterations = 1
    worse_factors = np.empty(iterations)
    for i in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        # gp = PoGPE(200)
        # gp = GPoGPE(200)
        gp = BCM(200)
        gp.fit(X_train, y_train)
        y_pred1, variances1 = gp.predict(X_test)
        mse1 = helpers.regression_score(y_test, y_pred1)

        # gp = GaussianProcess()
        # gp.fit(X_train, y_train)
        # y_pred2, variances2 = gp.predict(X_test)
        # mse2 = helpers.regression_score(y_test, y_pred2)
        # print(mse2)

        y_train = y_train.reshape(y_train.shape[0], 1)
        m = GPy.models.GPRegression(X_train, y_train)
        y_pred3, variances3 = m.predict(X_test)
        mse3 = helpers.regression_score(y_test, y_pred3)


        print("ensemble: {}, gpy: {}".format(mse1, mse3))
        
        worse_factor = mse1/mse3
        worse_factors[i] = worse_factor
        print("ensemble was {} times higher than GPy".format(worse_factor))

        vis.plot(X_train, y_train, X_test, y_pred3, variances3)
    print("worse factor average: {}".format(np.average(worse_factors)))

def classification_dummy_testing():
    ####################################################################################
    # Dummy testing - classification
    ####################################################################################

    gp = GaussianProcess()

    #X, y = datasets.make_circles(n_samples=12)
    # n_informative
    # n_redundant
    # n_repeated
    redundant = 0
    repeated = 0
    classes = 2
    clusters = 2
    informative = math.ceil(math.sqrt(classes*clusters))
    features = informative + repeated + redundant

    # classes * clusters <= 2**informative
    # informative + redundant + repeated < features
    X, y = datasets.make_classification(n_samples=200,
            n_features=features, 
            n_clusters_per_class=clusters,
            n_redundant=redundant, 
            n_repeated=repeated,
            n_informative=informative,
            n_classes=classes)

    from sklearn.cross_validation import StratifiedKFold
    # cv = StratifiedKFold(y, n_folds = 20)
    cv = StratifiedShuffleSplit(y, 1, test_size=20/len(X))
    accuracies = []
    for train_index, test_index in cv:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # LR
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = helpers.score(y_pred, y_test)
        print("LR F-score is: {}".format(accuracy))
        vis.plot_classes(X_test, y_test, y_pred)

        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = helpers.score(y_pred, y_test)
        print("RF F-score is: {}".format(accuracy))
        vis.plot_classes(X_test, y_test, y_pred)

        clf = SVC()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = helpers.score(y_pred, y_test)
        print("SVM F-score is: {}".format(accuracy))
        vis.plot_classes(X_test, y_test, y_pred)

        # GP
        gp.fit(X_train, y_train)
        y_pred = gp.predict(X_test)
        accuracy = helpers.score(y_pred, y_test)
        accuracies.append(accuracy)
        print("GP F-score is: {}".format(accuracy))
        vis.plot_classes(X_test, y_test, y_pred)
    # print("Average accuracy: {}".format(np.average(accuracies)))

def testGP(gp, features, labels, idx, n_iter=5):
    rem_idx = np.array(list(set(np.arange(features.shape[0])) - set(idx)))
    aurocs = []
    scores = []
    for i in range(n_iter):
        gp.fit(features[idx], labels[idx])
        means, var = gp.predict(features[rem_idx], keep_probs=True)
        auroc = helpers.roc_auc_score_multi(labels[rem_idx], means)
        score = helpers.score(labels[rem_idx], np.argmax(means, axis=0))
        print("This round: auroc: {}, f1: {}".format(auroc, score))
        aurocs.append(auroc)
        scores.append(score)
    return aurocs, scores

def dm_test_data():
    # Settings
    sampperclass1 = 20
    sampperclass2 = 20
    multisamp1 = 500
    multisamp2 = 10
    kfolds = 5
    activation = 'soft'

    # Make data
    X1 = np.random.multivariate_normal([-5, -5], [[1, 0], [0, 1]], sampperclass1)
    X2 = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], sampperclass2)
    C1 = np.random.multinomial(multisamp1, [0.1, 0.7, 0.2], sampperclass1)
    C2 = np.random.multinomial(multisamp2, [0.1, 0.2, 0.7], sampperclass2)

    # Concatenate data
    X = np.vstack((X1, X2))
    C = np.vstack((C1, C2))

    return X, C

def dir_mul():
    X, C = dm_test_data()
    dm = DirichletMultinomialRegression()
    dm.fit(X, C)
    pdb.set_trace()
    

def other_algos_bench(features, labels):
    ########################################## Compare other Algos ###########################################
    classifiers = [
            neighbors.KNeighborsClassifier(n_neighbors=5),                  
            LogisticRegression(),                                           
            # LogisticRegression(multi_class='multinomial', solver='lbfgs'), 
            RandomForestClassifier(),                                       
            # SVC()
            ]

    # X_train, X_test, y_train, y_test = train_test_split(features_sn, labels_simple, test_size = 0.1)
    # X_train, X_test = features_sn[train_idx], features_sn[test_idx]
    # y_train, y_test = labels_simple[train_idx], labels_simple[test_idx]
    # rf = RandomForestClassifier()
    # y_ = rf.fit(X_train, y_train).predict(X_test)
    # NOTE this doesn't work! can't roc_auc_score on discrete predictions
    # score = helpers.roc_auc_score_multi(y_test, y_)
    # print(score)

    # 10-fold cross-validation for all
    results = []
    for classifier in classifiers:
        results.append(cross_validate_algo(features, labels, 10, classifier))

    return results

    # # 10-fold cross-validation for all
    # for classifier in classifiers:
    #     cross_validate_algo(features, labels, 10, classifier)
    #########################################################################################################

def dm_vs_det_stats(dm_distrs, det_labels):
    """
    Compares the output of the Dirichlet Multinomial Regressor (its distribution of labels) with that of a set of deterministic results (including the argmax of a GP Classifier)
    """
    result_str = ""

    dm_argsorted = np.argsort(dm_distrs, axis=1)
    mismatch_idxs = np.where(dm_distrs.argmax(axis=1) != det_labels)
    rankings = np.array([np.where(x==y)[0][0] for x,y in zip(dm_argsorted[mismatch_idxs], det_labels[mismatch_idxs])])
    result_str += "For mismatches, from 2nd most probable to least probable compared to det_labels:\n"
    for i, cnt in enumerate(np.bincount(rankings)):
        result_str += "\t{} most likely occurrences: {}\t{}%\n".format(i, cnt, cnt/mismatch_idxs[0].shape[0]*100)

    match_count = np.sum((dm_distrs.argmax(axis=1) == det_labels))
    result_str += "Argmax of the dm distrs and the deterministic labels had: {} matches, i.e. {}%".format(match_count, match_count/det_labels.shape[0])

    print(result_str)
    return result_str

