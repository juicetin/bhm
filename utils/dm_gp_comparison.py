import numpy as np

from ML.gp.gp_gpy import GPyC

from utils.load_data import generate_dm_toy_ex
from ML.dir_mul.dirichlet_multinomial import DirichletMultinomialRegression
from ML.dir_mul.nicta.dirmultreg import dirmultreg_learn, dirmultreg_predict
from ML.gp.gp import GaussianProcess
from ML.dir_mul.dirichlet_multinomial import DirichletMultinomialRegression
import pdb

import utils.visualisation as vis

def normalise(data):
    return data/data.sum(axis=1)[:,np.newaxis]

def dm_vs_gp():
    X_train_c, X_test_c, X_train, X_test, C_train, C_test = generate_dm_toy_ex()   

    results = {}
    results['dm'] = {}
    results['gp'] = {}

    # Dirichlet multinomial stuff
    W = dirmultreg_learn(X_train, C_train, verbose=True, reg=0.1)
    EC, _ = dirmultreg_predict(X_test, W)

    C_test_norm = normalise(C_test)
    mean_err_dm1 = np.average(np.abs(EC-C_test_norm))
    print(mean_err_dm1)

    # GPy
    gpy = GPyC()
    gpy.fit(X_train, C_train.argmax(axis=1)[:,np.newaxis])
    gp_preds2, gp_vars2 = gpy.predict(X_test)
    mean_err_gp2 = np.average(np.abs(gp_preds2.argmax(axis=0) - C_test.argmax(axis=1)))
    print(mean_err_gp2)

    # gp = GaussianProcess()
    # gp.fit(X_train, C_train.argmax(axis=1))
    # gp_probs, gp_vars1 = gp.predict(C_train.argmax(axis=1), keep_probs=True)
    # gp_preds1 = gp_probs.argmax(axis=0)
    # mean_err_gp1 = np.average(np.abs(gp_preds1 - C_test.argmax(axis=1)))
    # print(mean_err_gp1)

    vis.plot_toy_data(np.concatenate((X_train_c, X_test_c)), np.concatenate((C_train, C_test)).argmax(axis=1), filename='toydataplot.pdf', display=False)

    pdb.set_trace()

    # dm = DirichletMultinomialRegression()
    # dm.fit(X_train, C_train)
    # preds = dm.predict(X_test)
    # mean_err_dm2 = np.average(np.abs(preds-C_test_norm))
    # print(mean_err_dm2)

    # # GP Stuff
    # gp = GaussianProcess()
    # gp.fit(X_train, C_train.argmax(axis=1))
    # gp_probs, gp_vars1 = gp.predict(C_train.argmax(axis=1), keep_probs=True)
    # gp_preds1 = gp_probs.argmax(axis=0)
    # mean_err_gp1 = np.average(np.abs(gp_preds1 - C_test.argmax(axis=1)))
    # print(mean_err_gp1)
